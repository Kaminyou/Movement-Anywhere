import enum


class SubtaskEnum(enum.Enum):
    DEPTH_ESTIMATION = 'depth_estimation'
    OPENOPSE = 'openpose'
    R_ESTIMATION = 'r_estimation'
    SVO_CONVRESION = 'svo_conversion'
    SVO_DEPTH_SENSING = 'svo_depth_sensing'
    TRACK_AND_EXTRACT = 'track_and_extract'
    TURN_TIME = 'turn_time'
    VIDEO_GENERATION_2D = 'video_generation_2d'
    VIDEO_GENERATION_3D = 'video_generation_3d'
