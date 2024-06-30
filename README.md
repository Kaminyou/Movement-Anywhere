![version](https://img.shields.io/badge/version-0.9.6-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Kaminyou/Gait-Anywhere/blob/main/LICENSE)
![linting workflow](https://github.com/Kaminyou/Gait-Anywhere/actions/workflows/main.yml/badge.svg)
# Gait-Anywhere
A distributed 2D video-based gait analysis platform

## Deployment
### Prerequisite
Please make sure you have `docker` and `docker-compose` installed. For k8s deployment, please make sure you have `kubectl`.


### Master node
1. Please execute `setup.sh` to download pretrained weights for several deep learning models. It will also check if all required docker images exist or not.
    ```bash
    $ ./setup.sh
    ```
2. Please modify `database/sql/create_user.sql` first to create accounts for default admin users.
3. Please create an `.env` file with the following format.
    ```env
    JWT_SECRET_KEY=...
    MYSQL_ROOT_PASSWORD=...
    SQLALCHEMY_DATABASE_URI=mysql+pymysql://root:<MYSQL_ROOT_PASSWORD>@db:3306/ndd
    DOCKER_SOCKET_PATH=... # usually at /var/run/docker.sock
    SYNC_FILE_SERVER_STORE_PATH=... # find somewhere large enough to save the results
    SYNC_FILE_SERVER_USER=...
    SYNC_FILE_SERVER_PASSWORD=...
    FOLDER_TO_STORE_TEMP_FILE_PATH=... # find somewhere large enough to save the temp files
    ```
4. Please make sure the model weights are present in the following path
    ```
    ./backend/algorithms/gait_basic/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin
    ./backend/algorithms/gait_basic/gait_study_semi_turn_time/weights/semi_vanilla_v2/gait-turn-time.pth
    ./backend/algorithms/gait_basic/depth_alg/weights/gait-depth-weight.pth
    ```
5. (Optional) Remove workers that you don't want them to run on the master node
6. Execute
    ```bash
    $ docker-compose up --build -d
    ```

### Client nodes (workers)
Configuration workers on client nodes is easy, please create a `docker-compose-SUFFIX.yml` file and add each worker's information
```yml
# copy x-common-variables: &common-variables block
# copy x-dind-worker-settings: &common-dind-worker-settings block
XXX-workerN:
    <<: *common-dind-worker-settings
    container_name: gait-anywhere-XXX-workerN
    environment:
      <<: *common-variables
      CELERY_WORKER: 'gait-worker'
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID'
      CUDA_VISIBLE_DEVICES: '0'  # change if needed
    command: celery --app algorithms.gait_basic.tasks.XXX_task worker -Q XXX_task_queue -n XXX-workerN@%h -c 1 --max-tasks-per-child=1 --without-heartbeat --loglevel=info --logfile=inference/logs/XXX-workerN.log
```
Then execute
```bash
$ docker-compose up --build -d
```

### Deploy with Kubernetes
Converting the docker compose YMAL to Kubernetes config is achievable by [kompose](https://kompose.io/)
```bash
$ kompose convert -f compose.yaml
```

## Active learning
To find out what are a better subset for turn time labeling among collected dataset, please follow the steps:
1. Please setup test environment as mentioned in `Unit test and integration test` section
2. Please execute
    ```bash
    $ docker exec -it gait-anywhere-test_env bash
    # in the container
    $ python3 tools/find_hard_examples.py --number 10  # will show top 10
    ```
3. The output will be like
    ```
    Number 1; certainty=0.3101; path=/data/XXX/out/3d/2024-05-04-1-14.mp4.npy
    Number 2; certainty=0.4529; path=/data/YYY/out/3d/2024-05-04-1-14.mp4.npy
    Number 3; certainty=0.4738; path=/data/ZZZ/out/3d/2024-05-04-1-14.mp4.npy
    ```

## Development Guide
### Get started
We provide a specific enviroment for developement. Please execute the following command after completing steps 1-4 in the previous section.
```
$ docker-compose -f docker-compose-dev.yml up --build -d
```
Then, you can access `frontend` by
```bash
$ docker exec -it gait-anywhere-frontend-dev bash
# in the container
$ yarn start
```

You can access `backend` by
```bash
$ docker exec -it gait-anywhere-backend-dev bash
# in the container
$ python3 app.py
```

### Unit test and integration test
Please set up all services according to the previous section, then
```bash
$ ./test_integration_setup.sh  # download test data
$ docker exec -it gait-anywhere-test_env bash
# in the container
$ pytest --cov backend/ .  # for unit tests
$ pytest --integration .  # for integration tests
```
Please note that the test script will not automatically delete anything created during the integration test (so as to enable debugging).
Before you set up the production services, please double check if you did clean up the database and the folder to store the files (at `SYNC_FILE_SERVER_STORE_PATH` in `.env`)

### Customization
#### Add new algorithms (models) or new data type
1. Please create a folder: `backend/algorithms/<YOUR_ALGORITHM_NAME>`.
2. Your folder should have a `__init__.py`.
3. In your analyzer file, add `from algorithms._analyzer import Analyzer`.
4. Create a class for your algorithm, which should inherit `Analyzer`.
    ```python
    class CustomizedAnalyzer(Analyzer):
        def __init__(
            self,
            ...,  # some pretrained weights' paths
        ):
            ...

        def run(
            self,
            request_uuid: str,
            data_root_dir: str,
            file_id: str,
            height: float = None,
            focal_length: float = None,
            ...,  # remove or add your args
        ) -> t.List[t.Dict[str, t.Any]]:
            # implementation
    ```
5. Make sure the return of `run` is in the format of `t.List[t.Dict[str, t.Any]]`.
6. Modify `MAPPING` in `backend/inference/config.py`. For example,
```python
YOUR_MODELS = {
    'your_model_name_1': CustomizedAnalyzer,
    'your_model_name_2': CustomizedAnalyzerV2,
}

MAPPING = {
    'data_type': YOUR_MODELS,
}

```
7. Finish. If you need to modify the input interface or anything else, please directly modify those files.
#### Implement your algorithms with Runner
Each analyzer should be executed by calling a pipeline (DAG) of subtask. Each subtask should be implemented as a Runner.
1. Please create a folder: `backend/algorithms/<YOUR_ALGORITHM_NAME>/tasks`. 
2. Add your subtask file, and import the Runner abstract `from algorithms._runner import Runner`
3. Your subtask runner should inherit `Runner`.
    ```python
    class CustomizedTaskRunner(Runner):
        def __init__(
            self,
            request_uuid: str,
            config: t.Dict[str, t.Any],
            data_synchronizer: DataSynchronizer,
            celery_task_id: str,
            update_state: t.Callable,
            result_hook: t.Optional[t.Dict] = None,
        ):
            self.request_uuid = request_uuid
            self.config = config
            self.file_id = self.config['file_id']
            self.data_synchronizer = data_synchronizer
            self.celery_task_id = celery_task_id
            self.update_state = update_state
            self.result_hook = result_hook

            # define your I/O here

        def fetch_data(self):
            # implementation

        def upload_data(self):
            # implementation

        def execute(self):
            # implementation

        def clear(self):
            shutil.rmtree(os.path.join(WORKER_WORKING_DIR_PATH, self.request_uuid))
    ```

## Important notes
Two essential docker images will be public available upon publication.

## Known issues
If a large mp4 video is uploading (e.g., 100MB with 5000 frames), a large RAM is needed to process the video (~35 GB).

## Citation
Pending
