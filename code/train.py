from config import CFG
from utils import train_yolo, update_data_yaml, check_and_download_dataset
from field_detection import FieldDetectionCFG
from player_detection import PlayerDetectionCFG
from puck_detection import PuckDetectionCFG
from jersey_recognition import JerseyRecognitionCFG


def main():
    if CFG.players_detection:
        print("Starting training for player detection model...")
        config = PlayerDetectionCFG()
        check_and_download_dataset(config)
        update_data_yaml(config)
        train_yolo(config)

    if CFG.puck_detection:
        # Код для тренировки модели детекции шайбы
        print("Starting training for puck detection model...")
        config = PuckDetectionCFG()
        check_and_download_dataset(config)
        update_data_yaml(config)
        train_yolo(config)

    if CFG.field_detection:
        # Код для тренировки модели детекции площадки
        print("Starting training for field detection model...")
        config = FieldDetectionCFG()
        check_and_download_dataset(config)
        update_data_yaml(config)
        train_yolo(config)

    if CFG.jersey_recognition:
        # Код для классификации номеров игроков
        print("Starting training for jersey classification model...")
        config = JerseyRecognitionCFG()
        check_and_download_dataset(config)
        update_data_yaml(config)
        train_yolo(config)

    if CFG.game_stage_classification:
        # Код для классификации стадии игры
        print("Starting training for game stage classification model...")
        # check_and_download_dataset(...)
        # update_data_yaml(...)
        # train_yolo(...)


if __name__ == "__main__":
    main()