import math
import os
import time

import mysql.connector
import pytest
import requests
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Celery scheduler
scheduler = Celery(
    'tasks',
    broker=os.environ.get('CELERY_BROKER_URL'),
    backend=os.environ.get('CELERY_BACKEND_URL'),
    result_expires=0,
    broker_transport_options={'visibility_timeout': 1382400},
)


@pytest.fixture(scope='module')
def api_token():
    json_data = {
        'account': 'general',
        'password': 'general',
    }
    response = requests.post('http://backend:5000/api/token/login', json=json_data, verify=False)
    return response.json().get('access_token')


@pytest.fixture(scope='module')
def submit_data(api_token):
    headers = {'Authorization': f'Bearer {api_token}'}
    svo_file_path = 'test_data/2024-05-04-1-14.svo'
    txt_file_path = 'test_data/2024-05-04-1-14.txt'
    with open(svo_file_path, 'rb') as f_svo, open(txt_file_path, 'rb') as f_txt:
        files = {
            'svoFile': f_svo,
            'txtFile': f_txt,
        }
        data = {
            'dataType': 'gait_svo_and_txt',
            'modelName': 'gait_svo::v1',
            'date': '2024-05-04',
            'description': 'test_integration_svo',
            'trialID': '2024-05-04-1-14',
        }
        response = requests.post(
            'http://backend:5000/api/user/upload/gait',
            headers=headers,
            files=files,
            data=data,
            verify=False,
        )
        response.raise_for_status()
        return response.json()


@pytest.mark.integration
def test_integration_3d_file_submission(submit_data):
    task_id, request_uuid = submit_data['task_id'], submit_data['request_uuid']
    assert os.path.exists(os.path.join('/data', request_uuid, 'input', '2024-05-04-1-14.svo'))
    assert os.path.exists(os.path.join('/data', request_uuid, 'input', '2024-05-04-1-14.txt'))

    # Wait for the task to complete
    task_obj = scheduler.AsyncResult(task_id)
    MAX_WAITING_TIME = 600
    for time_accum in range(0, MAX_WAITING_TIME, 10):
        if task_obj.ready():
            break
        time.sleep(10)
    else:
        pytest.fail(f'Task did not complete in expected time ({MAX_WAITING_TIME}) s')

    # Check output files
    for file_name in ['render-black-background.mp4', 'render.mp4', '2024-05-04-1-14-tt.pickle']:
        assert os.path.exists(os.path.join('/data', request_uuid, 'out', file_name))

    # Check output values
    connection = mysql.connector.connect(
        host='db',
        database='ndd',
        user='root',
        password=os.environ['MYSQL_ROOT_PASSWORD'],
    )
    try:
        cursor = connection.cursor()
        query = f"SELECT * FROM results WHERE requestUUID = '{request_uuid}';"
        cursor.execute(query)
        records = cursor.fetchall()
        output_gait_parameters = {row[3]: {
            'value': float(row[4]),
            'unit': row[6],
        } for row in records}

        expected_gait_parameters = {
            'stride length': {'value': 85.89533333333335, 'unit': 'cm'},
            'stride width': {'value': 30.146693333333296, 'unit': 'cm'},
            'stride time': {'value': 1.355, 'unit': 's'},
            'velocity': {'value': 0.633913899138991, 'unit': 'm/s'},
            'cadence': {'value': 44.280442804428, 'unit': '1/min'},
            'turn time': {'value': 1.71, 'unit': 's'},
        }

        assert len(output_gait_parameters) == len(expected_gait_parameters)
        assert set(output_gait_parameters.keys()) == set(expected_gait_parameters.keys())

        for key in output_gait_parameters:
            assert math.isclose(
                output_gait_parameters[key]['value'],
                expected_gait_parameters[key]['value'],
                rel_tol=1e-9,
            )
            assert output_gait_parameters[key]['unit'] == expected_gait_parameters[key]['unit']

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
