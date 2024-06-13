import os

from sqlalchemy.orm.session import Session

from models import ResultModel
from schemas.result import ResultSchema
from .config import data_and_model_map_to_class, get_focal_length_by_model_name


result_schema = ResultSchema()


def inference_gait(
    dataType: str,
    modelName: str,
    submitUUID: str,
    session: Session,
    trial_id: str,
    height: float,
    focal_length: float,
):
    try:
        model_focal_length = get_focal_length_by_model_name(modelName)
        analyzer = data_and_model_map_to_class(
            data_type=dataType,
            model_name=modelName,
        )(model_focal_length=model_focal_length)
    except Exception as e:
        print(e)
        raise ValueError(f'dataType={dataType} and modelName={modelName} not exist')

    results = analyzer.run(
        submit_uuid=submitUUID,
        data_root_dir=os.path.join('/data/', submitUUID),
        file_id=trial_id,
        height=height,
        focal_length=focal_length,
    )

    for result in results:
        form_date = result_schema.load({
            'requestUUID': submitUUID,
            'resultKey': result['key'],
            'resultValue': str(result['value']),
            'resultType': result['type'],
            'resultUnit': result['unit'],
        })
        resultObj = ResultModel(**form_date)
        session.add(resultObj)
        session.commit()
