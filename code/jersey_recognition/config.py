from utils.base_config import BaseConfig

class JerseyRecognitionCFG(BaseConfig):
    model = 'yolov8s'
    epochs = 50
    image_size = 1100
    batch_size = 10

    dataset_path = 'jersey_recognition'
    task_name = 'jersey_recognition'

    dataset_workspace = 'fastdeploy'
    dataset_project = '-923m4'
    dataset_version = 1
