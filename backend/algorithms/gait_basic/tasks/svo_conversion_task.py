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

SVO_EXPORT_RETRY = 2

SYNC_FILE_SERVER_URL = os.environ['SYNC_FILE_SERVER_URL']
SYNC_FILE_SERVER_PORT = os.environ['SYNC_FILE_SERVER_PORT']
SYNC_FILE_SERVER_USER = os.environ['SYNC_FILE_SERVER_USER']
SYNC_FILE_SERVER_PASSWORD = os.environ['SYNC_FILE_SERVER_PASSWORD']
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_BACKEND_URL = os.environ.get('CELERY_BACKEND_URL')
TASK_SYNC_URL = os.environ.get('TASK_SYNC_URL')

FOLDER_TO_STORE_TEMP_FILE_PATH = os.environ.get('FOLDER_TO_STORE_TEMP_FILE_PATH')
DOCKER_NETWORK = os.environ.get('DOCKER_NETWORK', None)

WORKER_WORKING_DIR_PATH = os.path.join('/root/data/', 'svo_conversion')

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


class SVOConversionTask(Runner):
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
        self.input_svo_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'input',
            f'{self.file_id}.svo',
        )
        self.input_svo_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'input',
            f'{self.file_id}.svo',
        )

        # output
        self.output_mp4_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'input',
            f'{self.file_id}.mp4',
        )
        self.output_mp4_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'input',
            f'{self.file_id}.mp4',
        )

        self.output_avi_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}.avi',
        )
        self.output_avi_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}.avi',
        )

    def fetch_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'fetching data'})
        self.data_synchronizer.download(
            src=self.input_svo_path_remote,
            des=self.input_svo_path_local,
        )

    def upload_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 100, 'stage': 'uploading data'})
        self.data_synchronizer.upload(
            src=self.output_mp4_path_local,
            des=self.output_mp4_path_remote,
        )
        self.data_synchronizer.upload(
            src=self.output_avi_path_local,
            des=self.output_avi_path_remote,
        )

    def execute(self):
        os.makedirs(
            os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out'), exist_ok=True,
        )
        # convert to avi
        retry = 0
        success = False
        while (retry < SVO_EXPORT_RETRY) and (not success):
            print(f'retry svo export time: {retry}')
            run_container(
                client=client,
                image='zed-env:latest',
                command=f'timeout 600 python3 /root/svo_export.py "{self.input_svo_path_local}" "{self.output_avi_path_local}" 0',  # noqa
                volumes={
                    BACKEND_FOLDER_PATH: {'bind': WORK_DIR, 'mode': 'rw'},
                    FOLDER_TO_STORE_TEMP_FILE_PATH: {'bind': '/root/data/', 'mode': 'rw'},
                },
                working_dir=WORK_DIR,
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=CUDA_VISIBLE_DEVICES.split(','),
                        capabilities=[['gpu']],
                    ),
                ],
            )
            retry += 1
            success = os.path.exists(self.output_avi_path_local)

        # avi to mp4 (rotate 90 clockwisely)
        run_container(
            client=client,
            image='zed-env:latest',
            command=f'python3 /root/avi_to_mp4.py --avi-path "{self.output_avi_path_local}" --mp4-path "{self.output_mp4_path_local}"',  # noqa
            volumes={
                BACKEND_FOLDER_PATH: {'bind': WORK_DIR, 'mode': 'rw'},
                FOLDER_TO_STORE_TEMP_FILE_PATH: {'bind': '/root/data/', 'mode': 'rw'},
            },
            working_dir=WORK_DIR,
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=CUDA_VISIBLE_DEVICES.split(','),
                    capabilities=[['gpu']],
                ),
            ],
        )

    def clear(self):
        shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid))


@app.task(bind=True, name='svo_conversion_task', queue='svo_conversion_task_queue')
def svo_conversion_task(self, submit_uuid: str, config: t.Dict[str, t.Any]):

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'svo_conversion_task_{submit_uuid}'
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

    runner = SVOConversionTask(
        submit_uuid=submit_uuid,
        config=config,
        data_synchronizer=data_synchronizer,
        celery_task_id=self.request.id,
        update_state=self.update_state,
    )

    self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'start'})
    runner.run()

    return True
