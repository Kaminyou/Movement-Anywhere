import os
import pickle
import shutil
import typing as t

from celery import Celery
from redis import Redis

from algorithms._runner import Runner
from algorithms.gait_basic.gait_study_semi_turn_time.inference import turn_time_simple_inference
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

WORKER_WORKING_DIR_PATH = os.path.join('/root/data/', 'turn_time')

BACKEND_FOLDER_PATH = os.environ['BACKEND_FOLDER_PATH']
WORK_DIR = '/root/backend'

app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
    result_expires=0,  # with the default setting, redis TTL is set to 1
    broker_transport_options={'visibility_timeout': 1382400},
)


class TurnTimeTaskRunner(Runner):
    def __init__(
        self,
        submit_uuid: str,
        config: t.Dict[str, t.Any],
        data_synchronizer: DataSynchronizer,
        celery_task_id: str,
        update_state: t.Callable,
        result_hook: t.Optional[t.Dict] = None,
    ):
        self.submit_uuid = submit_uuid
        self.config = config
        self.file_id = self.config['file_id']
        self.data_synchronizer = data_synchronizer
        self.celery_task_id = celery_task_id
        self.update_state = update_state
        self.result_hook = result_hook

        # input
        self.input_3dkeypoint_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            '3d',
            f'{self.file_id}.mp4.npy',
        )
        self.input_3dkeypoint_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            '3d',
            f'{self.file_id}.mp4.npy',
        )

        # output
        self.output_raw_turn_time_prediction_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-tt.pickle',
        )
        self.output_raw_turn_time_prediction_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-tt.pickle',
        )

    def fetch_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'fetching data'})
        self.data_synchronizer.download(
            src=self.input_3dkeypoint_path_remote,
            des=self.input_3dkeypoint_path_local,
        )

    def upload_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 100, 'stage': 'uploading data'})
        self.data_synchronizer.upload(
            src=self.output_raw_turn_time_prediction_path_local,
            des=self.output_raw_turn_time_prediction_path_remote,
        )

    def execute(self):
        tt, raw_tt_prediction = turn_time_simple_inference(
            turn_time_pretrained_path=self.config['turn_time_pretrained_path'],
            path_to_npz=self.input_3dkeypoint_path_local,
            return_raw_prediction=True,
        )

        with open(self.output_raw_turn_time_prediction_path_local, 'wb') as handle:
            pickle.dump(raw_tt_prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.result_hook is not None:
            self.result_hook['tt'] = tt

    def clear(self):
        shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid))


@app.task(bind=True, name='turn_time_task', queue='turn_time_task_queue')
def turn_time_task(self, submit_uuid: str, config: t.Dict[str, t.Any]):

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'turn_time_task_{submit_uuid}'
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

    result_hook = {'tt': -1}

    runner = TurnTimeTaskRunner(
        submit_uuid=submit_uuid,
        config=config,
        data_synchronizer=data_synchronizer,
        celery_task_id=self.request.id,
        update_state=self.update_state,
        result_hook=result_hook,
    )

    self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'start'})
    runner.run()

    return result_hook['tt']
