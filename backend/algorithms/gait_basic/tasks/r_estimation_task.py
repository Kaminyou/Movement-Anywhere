import os
import shutil
import typing as t

import pandas as pd
from celery import Celery
from redis import Redis

from algorithms._runner import Runner
from algorithms.gait_basic.utils.calculate import avg, replace_in_filenames
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

WORKER_WORKING_DIR_PATH = os.path.join('/root/data/', 'r_estimation')

BACKEND_FOLDER_PATH = os.environ['BACKEND_FOLDER_PATH']
WORK_DIR = '/root/backend'

app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
    result_expires=0,  # with the default setting, redis TTL is set to 1
    broker_transport_options={'visibility_timeout': 1382400},
)


class REstimationTaskRunner(Runner):
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
        self.input_csv_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-raw.csv',
        )
        self.input_csv_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}-raw.csv',
        )

        # output
        self.output_csv_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}.csv',
        )
        self.output_csv_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            f'{self.file_id}.csv',
        )

        self.output_zgait_folder_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.submit_uuid,
            'out',
            'zGait',
        )
        self.output_zgait_folder_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.submit_uuid,
            'out',
            'zGait',
        )

        if self.output_zgait_folder_local[-1] != '/':
            self.output_zgait_folder_local += '/'

        if self.output_zgait_folder_remote[-1] != '/':
            self.output_zgait_folder_remote += '/'

    def fetch_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'fetching data'})
        self.data_synchronizer.download(
            src=self.input_csv_path_remote,
            des=self.input_csv_path_local,
        )

    def upload_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 100, 'stage': 'uploading data'})
        self.data_synchronizer.upload(
            src=self.output_csv_path_local,
            des=self.output_csv_path_remote,
        )
        self.data_synchronizer.upload_folder(
            src_folder=self.output_zgait_folder_local,
            des_folder=self.output_zgait_folder_remote,
        )

    def execute(self):
        os.makedirs(
            os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out'), exist_ok=True,
        )

        shutil.copytree('algorithms/gait_basic/zGait/', self.output_zgait_folder_local)
        shutil.copyfile(
            self.input_csv_path_local,
            os.path.join(
                self.output_zgait_folder_local,
                'input',
                '2001-01-01-1',
                '2001-01-01-1-1.csv',
            ),
        )
        os.system(f'cd {self.output_zgait_folder_local} && Rscript gait_batch.R input/20010101.csv')
        shutil.copyfile(
            os.path.join(self.output_zgait_folder_local, 'output/2001-01-01-1/2001-01-01-1.csv'),
            self.output_csv_path_local,
        )
        replace_in_filenames(self.output_zgait_folder_local, '2001-01-01-1', self.file_id)

        # capture gait parameters
        df = pd.read_csv(self.output_csv_path_local, index_col=0)
        table = df.T['total'].T
        left_n = table['left.size']
        right_n = table['right.size']
        left_sl = table['left.stride.lt.mu']
        right_sl = table['right.stride.lt.mu']
        left_sw = table['left.stride.wt.mu']
        right_sw = table['right.stride.wt.mu']
        left_st = table['left.stride.t.mu']
        right_st = table['right.stride.t.mu']
        # tt = table['turn.t']
        cadence = table['cadence']
        velocity = table['velocity']

        sl = avg(left_sl, right_sl, left_n, right_n)
        sw = avg(left_sw, right_sw, left_n, right_n)
        st = avg(left_st, right_st, left_n, right_n)

        if self.result_hook is not None:
            self.result_hook['sl'] = sl
            self.result_hook['sw'] = sw
            self.result_hook['st'] = st
            self.result_hook['velocity'] = velocity
            self.result_hook['cadence'] = cadence

    def clear(self):
        shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid))


@app.task(bind=True, name='r_estimation_task', queue='r_estimation_task_queue')
def r_estimation_task(self, submit_uuid: str, config: t.Dict[str, t.Any]):

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'r_estimation_task_{submit_uuid}'
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

    result_hook = {
        'sl': -1,
        'sw': -1,
        'st': -1,
        'velocity': -1,
        'cadence': -1,
    }

    runner = REstimationTaskRunner(
        submit_uuid=submit_uuid,
        config=config,
        data_synchronizer=data_synchronizer,
        celery_task_id=self.request.id,
        update_state=self.update_state,
        result_hook=result_hook,
    )

    self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'start'})
    runner.run()

    return result_hook
