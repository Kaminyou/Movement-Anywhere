import os
import pickle
import shutil
import time
import typing as t

import pandas as pd
from celery.result import allow_join_result

from .._analyzer import Analyzer
from .utils.calculate import add_newline_if_missing, avg, fix_timestamp_file, replace_in_filenames
from .utils.docker_utils import run_container
from .utils.make_video import new_render
from .utils.track import (
    remove_non_target_person, set_zero_prob_for_keypoint_before_start_line,
)


BACKEND_FOLDER_PATH = os.environ['BACKEND_FOLDER_PATH']
WORK_DIR = '/root/backend'
START_LINE = 1820
SVO_EXPORT_RETRY = 2
DEPTH_SENSING_RETRY = 5
SYNC_FILE_SERVER_STORE_PATH = os.environ['SYNC_FILE_SERVER_STORE_PATH']

if os.environ.get('CELERY_WORKER', 'none') == 'gait-worker':

    import docker

    from .tasks.openpose_task import openpose_task
    from .tasks.svo_conversion_task import svo_conversion_task
    from .tasks.svo_depth_sensing_task import svo_depth_sensing_task
    from .tasks.track_and_extract_task import track_and_extract_task
    from .tasks.turn_time_task import turn_time_task
    from .tasks.r_estimation_task import r_estimation_task
    from .tasks.video_generation_3d_task import video_generation_3d_task

    CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
    client = docker.from_env(timeout=120)


class SVOGaitAnalyzer(Analyzer):
    def __init__(
        self,
        turn_time_pretrained_path: str = 'algorithms/gait_basic/gait_study_semi_turn_time/weights/semi_vanilla_v2/gait-turn-time.pth',  # noqa
        **kwargs,
    ):
        self.turn_time_pretrained_path = turn_time_pretrained_path

    def run(
        self,
        submit_uuid: str,
        data_root_dir: str,
        file_id: str,
        **kwargs,
    ) -> t.List[t.Dict[str, t.Any]]:

        def on_msg(*args, **kwargs):
            print(f'on_msg: {args}, {kwargs}')

        os.makedirs(os.path.join(data_root_dir, 'out'), exist_ok=True)
        os.makedirs(os.path.join(data_root_dir, 'out', '2d'), exist_ok=True)
        os.makedirs(os.path.join(data_root_dir, 'out', '3d'), exist_ok=True)
        os.makedirs(os.path.join(data_root_dir, 'video'), exist_ok=True)

        # input
        source_txt_path = os.path.join(data_root_dir, 'input', f'{file_id}.txt')

        # meta output
        meta_json_path = os.path.join(data_root_dir, 'out', f'{file_id}-json/')

        # meta output (for non-target person removing)
        meta_targeted_person_bboxes_path = os.path.join(data_root_dir, 'out', f'{file_id}-target_person_bboxes.pickle')  # noqa

        if not add_newline_if_missing(source_txt_path):
            print('add a new line to txt')

        # algorithm
        # step 1: svo conversion
        svo_config = {
            'file_id': file_id,
        }
        svo_conversion_task_instance = svo_conversion_task.delay(
            submit_uuid,
            svo_config,
        )
        while not svo_conversion_task_instance.ready():
            time.sleep(3)

        if svo_conversion_task_instance.failed():
            raise RuntimeError('SVO Conversion Task falied!')

        # step 2: openpose
        openpose_config = {
            'file_id': file_id,
        }
        openpose_task_instance = openpose_task.delay(
            submit_uuid,
            openpose_config,
        )
        while not openpose_task_instance.ready():
            time.sleep(3)

        if openpose_task_instance.failed():
            raise RuntimeError('Openpose Task falied!')

        # step 3: tracking and 3d lifting
        track_and_extract_config = {
            'file_id': file_id,
        }
        track_and_extract_task_instance = track_and_extract_task.delay(
            submit_uuid,
            track_and_extract_config,
        )
        while not track_and_extract_task_instance.ready():
            time.sleep(3)

        if track_and_extract_task_instance.failed():
            raise RuntimeError('Track and Extract Task falied!')

        # step 4: processing
        with open(meta_targeted_person_bboxes_path, 'rb') as handle:
            targeted_person_bboxes = pickle.load(handle)

        remove_non_target_person(meta_json_path, targeted_person_bboxes)

        # step 5: only allow after start line
        set_zero_prob_for_keypoint_before_start_line(
            json_path=meta_json_path,
            start_line=START_LINE,
        )

        # step 6: fix timestemp
        fix_timestamp_file(timestamp_file_path=source_txt_path, json_path=meta_json_path)

        # step 7: svo depth sensing
        svo_depth_sensing_config = {
            'file_id': file_id,
        }
        svo_depth_sensing_instance = svo_depth_sensing_task.delay(
            submit_uuid,
            svo_depth_sensing_config,
        )
        while not svo_depth_sensing_instance.ready():
            time.sleep(3)

        if svo_depth_sensing_instance.failed():
            raise RuntimeError('SVO Depth Sensing Task falied!')

        # step 8: run R to get gait parameters
        r_estimation_config = {
            'file_id': file_id,
        }
        r_estimation_instance = r_estimation_task.delay(
            submit_uuid,
            r_estimation_config,
        )
        while not r_estimation_instance.ready():
            time.sleep(3)

        if r_estimation_instance.failed():
            raise RuntimeError('R estimation falied!')
        
        sl = -1
        sw = -1
        st = -1
        cadence = -1
        velocity = -1
        with allow_join_result():
            try:
                gait_parameters = r_estimation_instance.get(on_message=on_msg, timeout=10)
                sl = gait_parameters['sl']
                sw = gait_parameters['sw']
                st = gait_parameters['st']
                cadence = gait_parameters['cadence']
                velocity = gait_parameters['velocity']
            except TimeoutError:
                print('Timeout!')

        # step 9: turn time
        turn_time_config = {
            'file_id': file_id,
            'turn_time_pretrained_path': self.turn_time_pretrained_path,
        }
        turn_time_task_instance = turn_time_task.delay(
            submit_uuid,
            turn_time_config,
        )
        while not turn_time_task_instance.ready():
            time.sleep(3)

        if turn_time_task_instance.failed():
            raise RuntimeError('Turn Time Task falied!')

        tt = -1
        with allow_join_result():
            try:
                tt = turn_time_task_instance.get(on_message=on_msg, timeout=10)
            except TimeoutError:
                print('Timeout!')

        # step 10: generate videos
        video_generation_3d_config = {
            'file_id': file_id,
        }
        video_generation_3d_task_instance = video_generation_3d_task.delay(
            submit_uuid,
            video_generation_3d_config,
        )
        while not video_generation_3d_task_instance.ready():
            time.sleep(3)

        if video_generation_3d_task_instance.failed():
            raise RuntimeError('Video Generation 3D Task falied!')

        return [
            {
                'key': 'stride length',
                'value': sl / 10,
                'unit': 'cm',
                'type': 'float',
            },
            {
                'key': 'stride width',
                'value': sw / 10,
                'unit': 'cm',
                'type': 'float',
            },
            {
                'key': 'stride time',
                'value': st / 1000,
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
