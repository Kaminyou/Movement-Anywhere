from models.subtask import SubtaskModel
from schemas.subtask import SubtaskSchema


subtask_schema = SubtaskSchema()


def register_subtask(session, request_uuid: str, subtask_instance, subtask_name):
    subtask_data = subtask_schema.load(
        {
            'requestUUID': request_uuid,
            'subtaskID': subtask_instance.id,
            'subtaskName': subtask_name,
        },
    )
    subtask_obj = SubtaskModel(**subtask_data)
    session.add(subtask_obj)
    session.commit()
