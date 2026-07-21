from utils.base_config import BaseConfig

class PuckDetectionCFG(BaseConfig):
    model = 'yolov8m'
    epochs = 50
    image_size = 1700
    batch_size = 8

    dataset_path = 'puck_detection'
    task_name = 'puck_detection'

    dataset_workspace = 'fastdeploy'
    dataset_project = '-w2x0g'
    dataset_version = 3
