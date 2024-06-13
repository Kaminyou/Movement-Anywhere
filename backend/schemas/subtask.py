from marshmallow import Schema, fields
from marshmallow.decorators import post_load
from marshmallow.validate import Length

from security import get_uuid


class SubtaskSchema(Schema):
    requestUUID = fields.Str(required=True, validate=Length(max=36))
    subtaskID = fields.Str(required=True, validate=Length(max=36))
    subtaskName = fields.Str(required=True, validate=Length(max=100))

    @post_load
    def add_uuid(self, data, **kwargs):
        data['subtaskUUID'] = get_uuid()
        return data
