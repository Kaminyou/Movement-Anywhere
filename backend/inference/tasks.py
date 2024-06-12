import os

from celery import Celery
from redis import Redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from enums.request import Status
from models import RequestModel

from .gait import inference_gait


CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_BACKEND_URL = os.environ.get('CELERY_BACKEND_URL')
TASK_SYNC_URL = os.environ.get('TASK_SYNC_URL')

app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
    result_expires=0,  # with the default setting, redis TTL is set to 1
    broker_transport_options={'visibility_timeout': 1382400},  # 16 days
)


@app.task(
    bind=True,
    name='inference_gait_task',
    queue='inference_gait_task_queue',
    default_retry_delay=60,
)
def inference_gait_task(self, submitUUID: str):
    print(submitUUID)
    """
    The gait inference process
    """

    redis = Redis.from_url(TASK_SYNC_URL)
    key = f'entry_task_{submitUUID}'
    if redis.exists(key):
        print(f'Skip this task since {key} exists')
        return True
    redis.set(key, 1)

    engine = create_engine(
        os.getenv(
            'SQLALCHEMY_DATABASE_URI',
        )
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    request_instance = session.query(RequestModel).filter_by(
        submitUUID=submitUUID).first()
    request_instance.status = Status.COMPUTING
    session.commit()

    dataType = request_instance.dataType
    modelName = request_instance.modelName
    trialID = request_instance.trialID
    height = request_instance.height
    focalLength = request_instance.focalLength

    try:
        inference_gait(
            dataType=dataType,
            modelName=modelName,
            submitUUID=submitUUID,
            session=session,
            trial_id=trialID,
            height=height,
            focal_length=focalLength,
        )

        request_instance = session.query(RequestModel).filter_by(
            submitUUID=submitUUID).first()
        request_instance.status = Status.DONE
        session.commit()

    except Exception as e:
        print(e)
        request_instance = session.query(RequestModel).filter_by(
            submitUUID=submitUUID).first()
        request_instance.status = Status.ERROR
        request_instance.statusInfo = str(e)[:75]
        session.commit()
        return True

    return True
