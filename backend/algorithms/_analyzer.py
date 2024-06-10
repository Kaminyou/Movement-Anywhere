import typing as t
from abc import ABC, abstractmethod


class Analyzer(ABC):
    def __init__(
        self,
        turn_time_pretrained_path: str,
    ):
        self.turn_time_pretrained_path = turn_time_pretrained_path

    @abstractmethod
    def run(
        self,
        submit_uuid: str,
        data_root_dir: str,
        file_id: str,
        height: float = None,
        focal_length: float = None,
    ) -> t.List[t.Dict[str, t.Any]]:
        pass
