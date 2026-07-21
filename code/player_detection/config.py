from utils.base_config import BaseConfig

class PlayerDetectionCFG(BaseConfig):
    model = 'yolov8s'
    epochs = 50
    image_size = 900
    batch_size = 24

    dataset_path = 'player_detection'
    task_name = 'player_detection'
    # classes = ['']
