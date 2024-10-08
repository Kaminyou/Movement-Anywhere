import typing as t

from db import db
from enums.request import Status


class RequestModel(db.Model):
    __tablename__ = "requests"

    id = db.Column(db.Integer, primary_key=True)
    account = db.Column(db.ForeignKey('users.account'), nullable=False)
    requestUUID = db.Column(db.CHAR(36), nullable=False, unique=True)

    # model and data info
    dataType = db.Column(db.String(100), nullable=False)
    modelName = db.Column(db.String(100), nullable=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())
    description = db.Column(db.String(200), nullable=False)
    trialID = db.Column(db.String(200), nullable=False)
    height = db.Column(db.Float, nullable=False, default=0.0)
    focalLength = db.Column(db.Float, nullable=False, default=1392.0)

    toShow = db.Column(db.Boolean, nullable=False, default=True)

    status = db.Column(db.Enum(Status), nullable=False, default=Status.INQUEUE)
    statusInfo = db.Column(db.String(80))
    dateUpload = db.Column(db.DateTime, default=db.func.current_timestamp())

    @classmethod
    def find_by_account(cls, account: str) -> t.List['RequestModel']:
        return cls.query.filter_by(account=account).order_by(RequestModel.dateUpload.desc()).all()

    @classmethod
    def find_by_account_and_sort_by_exp_date(cls, account: str) -> t.List['RequestModel']:
        return cls.query.filter_by(account=account).order_by(RequestModel.date.desc()).all()

    @classmethod
    def find_by_requestUUID(cls, requestUUID: str) -> 'RequestModel':
        return cls.query.filter_by(requestUUID=requestUUID).first()

    @classmethod
    def set_status_to_compute(cls, requestUUID: str) -> None:
        submission = cls.query.filter_by(requestUUID=requestUUID).first()
        submission.status = Status.COMPUTING
        db.session.commit()

    @classmethod
    def set_status_to_done(cls, requestUUID: str) -> None:
        submission = cls.query.filter_by(requestUUID=requestUUID).first()
        submission.status = Status.DONE
        db.session.commit()

    @classmethod
    def set_status_to_error(cls, requestUUID: str, error_msg: str = '') -> None:
        submission = cls.query.filter_by(requestUUID=requestUUID).first()
        submission.status = Status.ERROR
        submission.statusInfo = error_msg
        db.session.commit()

    @classmethod
    def is_download_link_exist(cls, requestUUID: str):
        submission = cls.query.filter_by(requestUUID=requestUUID).first()
        return submission.download is not None

    @classmethod
    def get_download_link(cls, requestUUID: str):
        submission = cls.query.filter_by(requestUUID=requestUUID).first()
        return submission.download

    def save_to_db(self) -> None:
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self) -> None:
        db.session.delete(self)
        db.session.commit()
