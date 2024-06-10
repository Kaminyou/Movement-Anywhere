import math
import os
import pickle
import shutil
import typing as t

from .._analyzer import Analyzer
from .gait_study_semi_turn_time.inference import turn_time_simple_inference
from .depth_alg.inference import depth_simple_inference
from .utils.make_video import new_render, count_frames
from .utils.track import (
    find_continuous_personal_bbox, load_mot_file
)
from .utils.docker_utils import run_container


BACKEND_FOLDER_PATH = os.environ['BACKEND_FOLDER_PATH']
WORK_DIR = '/root/backend'
START_LINE = 1820
SVO_EXPORT_RETRY = 2
DEPTH_SENSING_RETRY = 5
SYNC_FILE_SERVER_STORE_PATH = os.environ['SYNC_FILE_SERVER_STORE_PATH']

if os.environ.get('CELERY_WORKER', 'none') == 'gait-worker':

    import docker

    CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
    client = docker.from_env(timeout=120)


class Video2DGaitAnalyzer(Analyzer):
    def __init__(
        self,
        turn_time_pretrained_path: str = 'algorithms/gait_basic/gait_study_semi_turn_time/weights/semi_vanilla_v2/gait-turn-time.pth',  # noqa
        depth_pretrained_path: str = 'algorithms/gait_basic/depth_alg/weights/gait-depth-weight.pth',  # noqa
        model_focal_length: float = 1392.0,
        **kwargs,
    ):
        self.turn_time_pretrained_path = turn_time_pretrained_path
        self.depth_pretrained_path = depth_pretrained_path
        if math.isclose(model_focal_length, -1):
            raise ValueError('model focal length is not provided')
        self.model_focal_length = model_focal_length

    def run(
        self,
        data_root_dir: str,
        file_id: str,
        height: float,
        focal_length: float,
    ) -> t.List[t.Dict[str, t.Any]]:

        os.makedirs(os.path.join(data_root_dir, 'out'), exist_ok=True)
        os.makedirs(os.path.join(data_root_dir, 'out', '2d'), exist_ok=True)
        os.makedirs(os.path.join(data_root_dir, 'out', '3d'), exist_ok=True)
        os.makedirs(os.path.join(data_root_dir, 'video'), exist_ok=True)

        # input
        source_mp4_path = os.path.join(data_root_dir, 'input', f'{file_id}.mp4')

        # meta output (for non-target person removing)
        meta_mot_path = os.path.join(data_root_dir, 'out', f'{file_id}.mot.txt')
        meta_mp4_folder = os.path.join(data_root_dir, 'video')
        meta_mp4_path = os.path.join(data_root_dir, 'video', f'{file_id}.mp4')
        meta_targeted_person_bboxes_path = os.path.join(data_root_dir, 'out', f'{file_id}-target_person_bboxes.pickle')  # noqa

        # output
        output_2dkeypoint_folder = os.path.join(data_root_dir, 'out', '2d')
        output_3dkeypoint_folder = os.path.join(data_root_dir, 'out', '3d')
        output_3dkeypoint_path = os.path.join(data_root_dir, 'out', '3d', f'{file_id}.mp4.npy')
        meta_custom_dataset_path = os.path.join(data_root_dir, 'out', f'{file_id}-custom-dataset.npz')  # noqa
        output_raw_turn_time_prediction_path = os.path.join(data_root_dir, 'out', f'{file_id}-tt.pickle')  # noqa

        output_shown_mp4_path = os.path.join(data_root_dir, 'out', 'render.mp4')
        output_shown_black_background_mp4_path = os.path.join(data_root_dir, 'out', 'render-black-background.mp4')  # noqa

        # algorithm
        # tracking
        run_container(
            client=client,
            image='tracking-env:latest',
            command=(
                f'python3 /root/track.py '
                f'--source "{source_mp4_path}" '
                f'--yolo-model yolov8s.pt '
                f'--classes 0 --tracking-method deepocsort '
                f'--reid-model clip_market1501.pt '
                f'--save-mot --save-mot-path {meta_mot_path} --device cuda:0'
            ),
            volumes={
                BACKEND_FOLDER_PATH: {'bind': WORK_DIR, 'mode': 'rw'},
                SYNC_FILE_SERVER_STORE_PATH: {'bind': '/data', 'mode': 'rw'},
            },
            working_dir='/root',  # sync with the dry run during the building phase
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=CUDA_VISIBLE_DEVICES.split(','),
                    capabilities=[['gpu']],
                ),
            ],
        )
        mot_dict = load_mot_file(meta_mot_path)
        count = count_frames(source_mp4_path)
        targeted_person_ids, targeted_person_bboxes = find_continuous_personal_bbox(count, mot_dict)

        with open(meta_targeted_person_bboxes_path, 'wb') as handle:
            pickle.dump(targeted_person_bboxes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # old pipeline
        shutil.copyfile(source_mp4_path, meta_mp4_path)

        os.system(
            'cd algorithms/gait_basic/VideoPose3D && python3 quick_run.py '
            f'--mp4_video_folder "{meta_mp4_folder}" '
            f'--keypoint_2D_video_folder "{output_2dkeypoint_folder}" '
            f'--keypoint_3D_video_folder "{output_3dkeypoint_folder}" '
            f'--targeted-person-bboxes-path "{meta_targeted_person_bboxes_path}" '
            f'--custom-dataset-path "{meta_custom_dataset_path}"'
        )

        tt, raw_tt_prediction = turn_time_simple_inference(
            turn_time_pretrained_path=self.turn_time_pretrained_path,
            path_to_npz=output_3dkeypoint_path,
            return_raw_prediction=True,
        )

        with open(output_raw_turn_time_prediction_path, 'wb') as handle:
            pickle.dump(raw_tt_prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

        final_output, gait_parameters = depth_simple_inference(
            detectron_2d_single_person_keypoints_path=meta_custom_dataset_path,
            rendered_3d_single_person_keypoints_path=output_3dkeypoint_path,
            height=height,
            model_focal_length=self.model_focal_length,
            used_camera_focal_length=focal_length,
            depth_pretrained_path=self.depth_pretrained_path,
            turn_time_mask_path=output_raw_turn_time_prediction_path,
            device='cpu',
        )

        output_shown_mp4_path_temp = output_shown_mp4_path + '.tmp.mp4'
        new_render(
            video_path=source_mp4_path,
            detectron_custom_dataset_path=meta_custom_dataset_path,
            tt_pickle_path=output_raw_turn_time_prediction_path,
            output_video_path=output_shown_mp4_path_temp,
            draw_keypoint=True,
        )
        # browser mp4v encoding issue -> convert to h264
        os.system(f'ffmpeg -y -i {output_shown_mp4_path_temp} -movflags +faststart -vcodec libx264 -f mp4 {output_shown_mp4_path}')  # noqa
        os.system(f'rm {output_shown_mp4_path_temp}')

        output_shown_black_background_mp4_path_temp = output_shown_black_background_mp4_path + '.tmp.mp4'  # noqa
        new_render(
            video_path=source_mp4_path,
            detectron_custom_dataset_path=meta_custom_dataset_path,
            tt_pickle_path=output_raw_turn_time_prediction_path,
            output_video_path=output_shown_black_background_mp4_path_temp,
            draw_keypoint=True,
            draw_background=False,
        )
        os.system(f'ffmpeg -y -i {output_shown_black_background_mp4_path_temp} -movflags +faststart -vcodec libx264 -f mp4 {output_shown_black_background_mp4_path}')  # noqa
        os.system(f'rm {output_shown_black_background_mp4_path_temp}')

        sl = final_output['sl']
        sw = final_output['sw']
        st = final_output['st']
        velocity = final_output['v']
        cadence = final_output['c']

        return [
            {
                'key': 'stride length',
                'value': sl,
                'unit': 'cm',
                'type': 'float',
            },
            {
                'key': 'stride width',
                'value': sw,
                'unit': 'cm',
                'type': 'float',
            },
            {
                'key': 'stride time',
                'value': st,
                'unit': 's',
                'type': 'float',
            },
            {
                'key': 'velocity',
                'value': velocity,
                'unit': 'm/s',
                'type': 'float',
            },
            {
                'key': 'cadence',
                'value': cadence,
                'unit': '1/min',
                'type': 'float',
            },
            {
                'key': 'turn time',
                'value': tt,
                'unit': 's',
                'type': 'float',
            },
        ]
