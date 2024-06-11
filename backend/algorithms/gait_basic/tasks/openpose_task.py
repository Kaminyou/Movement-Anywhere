import os
import pickle
import shutil
import typing as t

from celery import Celery
from redis import Redis

from algorithms._runner import Runner
from algorithms.gait_basic.utils.docker_utils import run_container
from algorithms.gait_basic.utils.make_video import count_frames
from algorithms.gait_basic.utils.track import find_continuous_personal_bbox, load_mot_file
from settings import SYNC_FILE_SERVER_RESULT_PATH
from utils.synchronizer import DataSynchronizer


SYNC_FILE_SERVER_URL = os.environ['SYNC_FILE_SERVER_URL']
SYNC_FILE_SERVER_PORT = os.environ['SYNC_FILE_SERVER_PORT']
SYNC_FILE_SERVER_USER = os.environ['SYNC_FILE_SERVER_USER']
SYNC_FILE_SERVER_PASSWORD = os.environ['SYNC_FILE_SERVER_PASSWORD']
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_BACKEND_URL = os.environ.get('CELERY_BACKEND_URL')
TASK_SYNC_URL = os.environ.get('TASK_SYNC_URL')

FOLDER_TO_STORE_TEMP_FILE_PATH = os.environ.get('FOLDER_TO_STORE_TEMP_FILE_PATH')
DOCKER_NETWORK = os.environ.get('DOCKER_NETWORK', None)

WORKER_WORKING_DIR_PATH = os.path.join('/root/data/', 'openpose')

BACKEND_FOLDER_PATH = os.environ['BACKEND_FOLDER_PATH']
WORK_DIR = '/root/backend'

app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
    result_expires=0,  # with the default setting, redis TTL is set to 1
    broker_transport_options={'visibility_timeout': 1382400},
)

if os.environ.get('CELERY_WORKER', 'none') == 'gait-worker':

    import docker

    CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
    client = docker.from_env(timeout=120)


class OpenposeTask(Runner):
    def __init__(
        self,
        submit_uuid: str,
        config: t.Dict[str, t.Any],
        data_synchronizer: DataSynchronizer,
        celery_task_id: str,
        update_state: t.Callable,
    ):
        self.submit_uuid = submit_uuid
        self.config = config
        self.file_id = self.config['file_id']
        self.data_synchronizer = data_synchronizer
        self.celery_task_id = celery_task_id
        self.update_state = update_state

        # input
        self.input_avi_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}.avi',
        )
        self.input_avi_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}.avi',
        )

        # output
        self.output_keypoints_avi_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-keypoints.avi',
        )
        self.output_keypoints_avi_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-keypoints.avi',
        )

        self.output_json_folder_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-json/',
        )
        self.output_json_folder_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-json/',
        )

        if self.output_json_folder_local[-1] != '/':
            self.output_json_folder_local += '/'

        if self.output_json_folder_remote[-1] != '/':
            self.output_json_folder_remote += '/'

    def fetch_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'fetching data'})
        self.data_synchronizer.download(
            src=self.input_avi_path_remote,
            des=self.input_avi_path_local,
        )

    def upload_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 100, 'stage': 'uploading data'})
        self.data_synchronizer.upload(
            src=self.output_keypoints_avi_path_local,
            des=self.output_keypoints_avi_path_remote,
        )
        self.data_synchronizer.upload_folder(
            src_folder=self.output_json_folder_local,
            des_folder=self.output_json_folder_remote,
        )

    def execute(self):
        # openpose
        run_container(
            client=client,
            image='openpose-env:latest',
            command=(
                f'./build/examples/openpose/openpose.bin '
                f'--video {self.input_avi_path_local} --write-video {self.output_keypoints_avi_path_local} '
                f'--write-json {self.output_json_folder_local} --frame_rotate 270 --camera_resolution 1920x1080 '  # noqa
                f'--display 0'
            ),
            volumes={
                BACKEND_FOLDER_PATH: {'bind': WORK_DIR, 'mode': 'rw'},
                FOLDER_TO_STORE_TEMP_FILE_PATH: {'bind': '/root/data/', 'mode': 'rw'},
            },
            working_dir='/openpose',
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=CUDA_VISIBLE_DEVICES.split(','),
                    capabilities=[['gpu']],
                ),
            ],
        )

    def clear(self):
        shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid))


@app.task(bind=True, name='openpose_task', queue='openpose_task_queue')
def openpose_task(self, submit_uuid: str, config: t.Dict[str, t.Any]):

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'openpose_task_{submit_uuid}'
    if redis.exists(key):
        print(f'Skip this task since {key} exists')
        return True
    redis.set(key, 1)

    data_synchronizer = DataSynchronizer(
        url=SYNC_FILE_SERVER_URL,
        port=SYNC_FILE_SERVER_PORT,
        user=SYNC_FILE_SERVER_USER,
        password=SYNC_FILE_SERVER_PASSWORD,
    )

    runner = OpenposeTask(
        submit_uuid=submit_uuid,
        config=config,
        data_synchronizer=data_synchronizer,
        celery_task_id=self.request.id,
        update_state=self.update_state,
    )

    self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'start'})
    runner.run()

    return True
