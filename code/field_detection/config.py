from utils.base_config import BaseConfig

class FieldDetectionCFG(BaseConfig):
    model = 'yolov8m-seg'
    epochs = 50
    image_size = 900
    batch_size = 24

    dataset_path = 'field_detection'
    dataset_roboflow_id = 'bAfPu00y1F'
    task_name = 'field_detection'
    # classes = ['']

    dataset_workspace = 'fastdeploy'
    dataset_project = '-v9r9n'
    dataset_version = 3

