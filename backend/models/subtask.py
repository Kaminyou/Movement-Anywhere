import typing as t

from db import db


class SubtaskModel(db.Model):
    __tablename__ = 'subtasks'

    id = db.Column(db.Integer, primary_key=True)
    requestUUID = db.Column(db.ForeignKey('requests.requestUUID'), nullable=False)  # for server
    subtaskUUID = db.Column(db.CHAR(36), nullable=False)  # for server
    subtaskID = db.Column(db.CHAR(36), nullable=False)  # for celery
    subtaskName = db.Column(db.String(100), nullable=False)

    createTime = db.Column(db.DateTime, default=db.func.current_timestamp())

    @classmethod
    def find_by_requestUUID(cls, requestUUID: str) -> t.List['SubtaskModel']:
        return cls.query.filter_by(requestUUID=requestUUID).all()

    @classmethod
    def find_by_subtaskUUID(cls, subtaskUUID: str) -> 'SubtaskModel':
        return cls.query.filter_by(subtaskUUID=subtaskUUID).first()

    def save_to_db(self) -> None:
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self) -> None:
        db.session.delete(self)
        db.session.commit()
