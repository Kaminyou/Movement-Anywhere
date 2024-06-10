import typing as t
import os
import shutil
import pickle

from celery import Celery
from redis import Redis

from algorithms._runner import Runner
from algorithms.gait_basic.utils.docker_utils import run_container
from algorithms.gait_basic.utils.track import (
    find_continuous_personal_bbox, load_mot_file
)
from algorithms.gait_basic.utils.make_video import count_frames
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

WORKER_WORKING_DIR_PATH = os.path.join('/root/data/', 'track_and_extract')

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


class TrackAndExtractTaskRunner(Runner):
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
        self.input_mp4_path_remote = os.path.join(SYNC_FILE_SERVER_RESULT_PATH, self.submit_uuid, 'input', f'{self.file_id}.mp4')
        self.input_mp4_path_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'input', f'{self.file_id}.mp4')

        # meta
        self.meta_mp4_folder_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'video')
        self.meta_mp4_path_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'video', f'{self.file_id}.mp4')
        
        # output
        self.output_mot_path_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', f'{self.file_id}.mot.txt')
        self.output_mot_path_remote = os.path.join(SYNC_FILE_SERVER_RESULT_PATH, self.submit_uuid, 'out', f'{self.file_id}.mot.txt')

        self.output_targeted_person_bboxes_path_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', f'{self.file_id}-target_person_bboxes.pickle')  # noqa
        self.output_targeted_person_bboxes_path_remote = os.path.join(SYNC_FILE_SERVER_RESULT_PATH, self.submit_uuid, 'out', f'{self.file_id}-target_person_bboxes.pickle')  # noqa

        self.output_2dkeypoint_folder_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', '2d')
        self.output_2dkeypoint_folder_remote = os.path.join(SYNC_FILE_SERVER_RESULT_PATH, self.submit_uuid, 'out', '2d')

        self.output_3dkeypoint_folder_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', '3d')
        self.output_3dkeypoint_folder_remote = os.path.join(SYNC_FILE_SERVER_RESULT_PATH, self.submit_uuid, 'out', '3d')

        self.output_custom_dataset_path_local = os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', f'{self.file_id}-custom-dataset.npz')  # noqa
        self.output_custom_dataset_path_remote = os.path.join(SYNC_FILE_SERVER_RESULT_PATH, self.submit_uuid, 'out', f'{self.file_id}-custom-dataset.npz')  # noqa

        if self.output_2dkeypoint_folder_local[-1] != '/':
            self.output_2dkeypoint_folder_local += '/'

        if self.output_2dkeypoint_folder_remote[-1] != '/':
            self.output_2dkeypoint_folder_remote += '/'
        
        if self.output_3dkeypoint_folder_local[-1] != '/':
            self.output_3dkeypoint_folder_local += '/'

        if self.output_3dkeypoint_folder_remote[-1] != '/':
            self.output_3dkeypoint_folder_remote += '/'

    def fetch_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'fetching data'})
        self.data_synchronizer.download(
            src=self.input_mp4_path_remote,
            des=self.input_mp4_path_local,
        )

    def upload_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 100, 'stage': 'uploading data'})
        self.data_synchronizer.upload(
            src=self.output_mot_path_local,
            des=self.output_mot_path_remote,
        )
        self.data_synchronizer.upload(
            src=self.output_targeted_person_bboxes_path_local,
            des=self.output_targeted_person_bboxes_path_remote,
        )
        self.data_synchronizer.upload(
            src=self.output_custom_dataset_path_local,
            des=self.output_custom_dataset_path_remote,
        )
        self.data_synchronizer.upload_folder(
            src_folder=self.output_2dkeypoint_folder_local,
            des_folder=self.output_2dkeypoint_folder_remote,
        )
        self.data_synchronizer.upload_folder(
            src_folder=self.output_3dkeypoint_folder_local,
            des_folder=self.output_3dkeypoint_folder_remote,
        )

    def execute(self):
        # tracking
        run_container(
            client=client,
            image='tracking-env:latest',
            command=(
                f'python3 /root/track.py '
                f'--source "{self.input_mp4_path_local}" '
                f'--yolo-model yolov8s.pt '
                f'--classes 0 --tracking-method deepocsort '
                f'--reid-model clip_market1501.pt '
                f'--save-mot --save-mot-path {self.output_mot_path_local} --device cuda:0'
            ),
            volumes={
                BACKEND_FOLDER_PATH: {'bind': WORK_DIR, 'mode': 'rw'},
                FOLDER_TO_STORE_TEMP_FILE_PATH: {'bind': '/root/data/', 'mode': 'rw'},
            },
            working_dir='/root',  # sync with the dry run during the building phase
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=CUDA_VISIBLE_DEVICES.split(','),
                    capabilities=[['gpu']],
                ),
            ],
        )

        os.makedirs(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out'), exist_ok=True)
        os.makedirs(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', '2d'), exist_ok=True)
        os.makedirs(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'out', '3d'), exist_ok=True)
        os.makedirs(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid, 'video'), exist_ok=True)

        mot_dict = load_mot_file(self.output_mot_path_local)
        count = count_frames(self.input_mp4_path_local)
        targeted_person_ids, targeted_person_bboxes = find_continuous_personal_bbox(count, mot_dict)

        with open(self.output_targeted_person_bboxes_path_local, 'wb') as handle:
            pickle.dump(targeted_person_bboxes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # old pipeline
        shutil.copyfile(self.input_mp4_path_local, self.meta_mp4_path_local)
        os.system(
            'cd algorithms/gait_basic/VideoPose3D && python3 quick_run.py '
            f'--mp4_video_folder "{self.meta_mp4_folder_local}" '
            f'--keypoint_2D_video_folder "{self.output_2dkeypoint_folder_local}" '
            f'--keypoint_3D_video_folder "{self.output_3dkeypoint_folder_local}" '
            f'--targeted-person-bboxes-path "{self.output_targeted_person_bboxes_path_local}" '
            f'--custom-dataset-path "{self.output_custom_dataset_path_local}"'
        )

    def clear(self):
        shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.submit_uuid))


@app.task(bind=True, name='track_and_extract_task', queue='track_and_extract_task_queue')
def track_and_extract_task(self, submit_uuid: str, config: t.Dict[str, t.Any]):

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'track_and_extract_task_{submit_uuid}'
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

    runner = TrackAndExtractTaskRunner(
        submit_uuid=submit_uuid,
        config=config,
        data_synchronizer=data_synchronizer,
        celery_task_id=self.request.id,
        update_state=self.update_state,
    )

    self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'start'})
    runner.run()

    return True
