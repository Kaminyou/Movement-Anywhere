import typing as t

from algorithms._analyzer import Analyzer
from algorithms.gait_basic.main import SVOGaitAnalyzer, Video2DGaitAnalyzer


GAIT_SVO_MODELS_SIMPLE = {
    'gait_svo::v1': SVOGaitAnalyzer,
}

GAIT_2D_MODELS_SIMPLE = {
    'gait_2d::v1': Video2DGaitAnalyzer,
}

MAPPING = {
    'gait_svo_and_txt': GAIT_SVO_MODELS_SIMPLE,
    'gait_mp4': GAIT_2D_MODELS_SIMPLE,
}


def get_data_types() -> t.List[str]:
    data_types = []
    for k, _ in MAPPING.items():
        data_types.append(k)
    return data_types


def get_model_names(data_type: str) -> t.List[str]:
    models = MAPPING[data_type]
    model_names = []
    for k, _ in models.items():
        model_names.append(k)
    return model_names


def data_and_model_map_to_class(data_type: str, model_name: str) -> Analyzer:
    return MAPPING[data_type][model_name]
