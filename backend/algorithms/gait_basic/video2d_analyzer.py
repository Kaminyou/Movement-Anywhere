import math
import os
import pickle
import shutil
import time
import typing as t

from celery.result import allow_join_result

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
    from .tasks.track_and_extract_task import track_and_extract_task
    from .tasks.turn_time_task import turn_time_task
    from .tasks.depth_estimation_task import depth_estimation_task
    from .tasks.video_generation_task import video_generation_task

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
        submit_uuid: str,
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
        track_and_extract_config = {
            'file_id': file_id,
        }
        track_and_extract_task_instance = track_and_extract_task.delay(submit_uuid, track_and_extract_config)
        while not track_and_extract_task_instance.ready():
            time.sleep(3)

        if track_and_extract_task_instance.failed():
            raise RuntimeError('Track and Extract Task falied!')

        turn_time_config = {
            'file_id': file_id,
            'turn_time_pretrained_path': self.turn_time_pretrained_path,
        }
        turn_time_task_instance = turn_time_task.delay(submit_uuid, turn_time_config)
        while not turn_time_task_instance.ready():
            time.sleep(3)

        if turn_time_task_instance.failed():
            raise RuntimeError('Turn Time Task falied!')

        tt = -1
        with allow_join_result():
            def on_msg(*args, **kwargs):
                print(f'on_msg: {args}, {kwargs}')
            try:
                tt = turn_time_task_instance.get(on_message=on_msg, timeout=10)
            except TimeoutError:
                print('Timeout!')
        
        final_output = {}
        gait_parameters = []
        depth_estimation_config = {
            'file_id': file_id,
            'height': height,
            'model_focal_length': self.model_focal_length,
            'focal_length': focal_length,
            'depth_pretrained_path': self.depth_pretrained_path,
        }
        depth_estimation_task_instance = depth_estimation_task.delay(submit_uuid, depth_estimation_config)
        while not depth_estimation_task_instance.ready():
            time.sleep(3)

        if depth_estimation_task_instance.failed():
            raise RuntimeError('Depth Estimation Task falied!')

        with allow_join_result():
            def on_msg(*args, **kwargs):
                print(f'on_msg: {args}, {kwargs}')
            try:
                final_output, gait_parameters = depth_estimation_task_instance.get(on_message=on_msg, timeout=10)
            except TimeoutError:
                print('Timeout!')

        video_generation_config = {
            'file_id': file_id,
        }
        video_generation_task_instance = video_generation_task.delay(submit_uuid, video_generation_config)
        while not video_generation_task_instance.ready():
            time.sleep(3)

        if video_generation_task_instance.failed():
            raise RuntimeError('Video Generation Task falied!')

        sl = final_output.get('sl', -1)
        sw = final_output.get('sw', -1)
        st = final_output.get('st', -1)
        velocity = final_output.get('v', -1)
        cadence = final_output.get('c', -1)

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
