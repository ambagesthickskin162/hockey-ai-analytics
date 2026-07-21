from typing import List

class BaseConfig:
    model:str = "yolov8s"
    batch_size:int = 8
    image_size:int = 640
    epochs:int = 50

    dataset_path: str = None
    dataset_roboflow_id: str = None
    task_name: str = None
    classes: List[str] = None

    dataset_workspace: str = 'fastdeploy'
    dataset_project: str = None
    dataset_version: int = 1
