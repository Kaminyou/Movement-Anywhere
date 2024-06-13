import os
import shutil
import typing as t

from celery import Celery
from redis import Redis

from algorithms._runner import Runner
from algorithms.gait_basic.utils.docker_utils import run_container
from algorithms.gait_basic.utils.make_video import new_render
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

WORKER_WORKING_DIR_PATH = os.path.join('/root/data/', 'video_generation_3d')

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


class VideoGeneration3DTaskRunner(Runner):
    def __init__(
        self,
        request_uuid: str,
        config: t.Dict[str, t.Any],
        data_synchronizer: DataSynchronizer,
        celery_task_id: str,
        update_state: t.Callable,
    ):
        self.request_uuid = request_uuid
        self.config = config
        self.file_id = self.config['file_id']
        self.data_synchronizer = data_synchronizer
        self.celery_task_id = celery_task_id
        self.update_state = update_state

        # input
        self.input_mp4_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'input',
            f'{self.file_id}.mp4',
        )
        self.input_mp4_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'input',
            f'{self.file_id}.mp4',
        )

        self.input_json_folder_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-json/',
        )
        self.input_json_folder_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-json/',
        )

        self.input_targeted_person_bboxes_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-target_person_bboxes.pickle',
        )
        self.input_targeted_person_bboxes_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-target_person_bboxes.pickle',
        )

        self.input_custom_dataset_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-custom-dataset.npz',
        )
        self.input_custom_dataset_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-custom-dataset.npz',
        )

        self.input_raw_turn_time_prediction_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-tt.pickle',
        )
        self.input_raw_turn_time_prediction_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-tt.pickle',
        )

        # output
        self.output_shown_mp4_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            'render.mp4',
        )
        self.output_shown_mp4_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'out',
            'render.mp4',
        )

        self.output_shown_black_background_mp4_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            'render-black-background.mp4',
        )
        self.output_shown_black_background_mp4_path_remote = os.path.join(
            SYNC_FILE_SERVER_RESULT_PATH,
            self.request_uuid,
            'out',
            'render-black-background.mp4',
        )

        # meta
        self.meta_rendered_mp4_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-rendered.mp4',
        )
        self.meta_rendered_black_background_mp4_path_local = os.path.join(
            WORKER_WORKING_DIR_PATH,
            self.request_uuid,
            'out',
            f'{self.file_id}-rendered-black-background.mp4',
        )
        self.output_shown_mp4_path_temp_local = self.output_shown_mp4_path_local + '.tmp.mp4'
        self.output_shown_black_background_mp4_path_temp_local = self.output_shown_black_background_mp4_path_local + '.tmp.mp4'  # noqa

        if self.input_json_folder_remote[-1] != '/':
            self.input_json_folder_remote += '/'

        if self.input_json_folder_local[-1] != '/':
            self.input_json_folder_local += '/'

    def fetch_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'fetching data'})
        self.data_synchronizer.download(
            src=self.input_mp4_path_remote,
            des=self.input_mp4_path_local,
        )
        self.data_synchronizer.download_folder(
            src_folder=self.input_json_folder_remote,
            des_folder=self.input_json_folder_local,
        )
        self.data_synchronizer.download(
            src=self.input_targeted_person_bboxes_path_remote,
            des=self.input_targeted_person_bboxes_path_local,
        )
        self.data_synchronizer.download(
            src=self.input_custom_dataset_path_remote,
            des=self.input_custom_dataset_path_local,
        )
        self.data_synchronizer.download(
            src=self.input_raw_turn_time_prediction_path_remote,
            des=self.input_raw_turn_time_prediction_path_local,
        )

    def upload_data(self):
        self.update_state(state='PROGRESS', meta={'progress': 100, 'stage': 'uploading data'})
        self.data_synchronizer.upload(
            src=self.output_shown_mp4_path_local,
            des=self.output_shown_mp4_path_remote,
        )
        self.data_synchronizer.upload(
            src=self.output_shown_black_background_mp4_path_local,
            des=self.output_shown_black_background_mp4_path_remote,
        )

    def execute(self):
        run_container(
            client=client,
            image='zed-env:latest',
            command=(
                f'python3 /root/result_render.py --mp4-path "{self.input_mp4_path_local}" '
                f'--json-path "{self.input_json_folder_local}" '
                f'--targeted-person-bboxes-path "{self.input_targeted_person_bboxes_path_local}" '
                f'--rendered-mp4-path "{self.meta_rendered_mp4_path_local}"'
            ),
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

        # render_removed_result but black backgound
        run_container(
            client=client,
            image='zed-env:latest',
            command=(
                f'python3 /root/result_render.py --mp4-path "{self.input_mp4_path_local}" '
                f'--json-path "{self.input_json_folder_local}" '
                f'--targeted-person-bboxes-path "{self.input_targeted_person_bboxes_path_local}" '
                f'--rendered-mp4-path "{self.meta_rendered_black_background_mp4_path_local}" '
                f'--draw-all-keypoints --draw-black-background'
            ),
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

        # (openpose + box) + turning; (openpose + box) is on video_path
        new_render(
            video_path=self.meta_rendered_mp4_path_local,
            detectron_custom_dataset_path=self.input_custom_dataset_path_local,
            tt_pickle_path=self.input_raw_turn_time_prediction_path_local,
            output_video_path=self.output_shown_mp4_path_temp_local,
            draw_keypoint=False,
        )
        # browser mp4v encoding issue -> convert to h264
        os.system(f'ffmpeg -y -i {self.output_shown_mp4_path_temp_local} -movflags +faststart -vcodec libx264 -f mp4 {self.output_shown_mp4_path_local}')  # noqa

        # for black background
        new_render(
            video_path=self.meta_rendered_black_background_mp4_path_local,
            detectron_custom_dataset_path=self.input_custom_dataset_path_local,
            tt_pickle_path=self.input_raw_turn_time_prediction_path_local,
            output_video_path=self.output_shown_black_background_mp4_path_temp_local,
            draw_keypoint=False,
        )
        os.system(f'ffmpeg -y -i {self.output_shown_black_background_mp4_path_temp_local} -movflags +faststart -vcodec libx264 -f mp4 {self.output_shown_black_background_mp4_path_local}')  # noqa

    def clear(self):
        shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.request_uuid))


@app.task(bind=True, name='video_generation_3d_task', queue='video_generation_3d_task_queue')
def video_generation_3d_task(self, request_uuid: str, config: t.Dict[str, t.Any]):

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'video_generation_3d_task_{request_uuid}'
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

    runner = VideoGeneration3DTaskRunner(
        request_uuid=request_uuid,
        config=config,
        data_synchronizer=data_synchronizer,
        celery_task_id=self.request.id,
        update_state=self.update_state,
    )

    self.update_state(state='PROGRESS', meta={'progress': 0, 'stage': 'start'})
    runner.run()

    return True
