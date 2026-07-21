import cv2
import numpy as np
from ultralytics import YOLO
import concurrent.futures


# Подгрузка весов моделей
field_det_model = YOLO('weights/field_detection.pt')
player_det_model = YOLO('weights/player_detection.pt')
puck_det_model = YOLO('weights/puck_detection.pt')
jersey_det_model = YOLO('weights/jersey_detection.pt')

def annotate_image(image: np.ndarray) -> np.ndarray:
    '''Функци разметки кадра'''
    
    # детекция всеми моделями с параллелизмом по потокам
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_field_detection = executor.submit(field_det_model.track, source=image, conf=0.30)
        future_player_detection = executor.submit(player_det_model.track, source=image, conf=0.25)
        future_puck_detection = executor.submit(puck_det_model.predict, source=image, conf=0.40)
        future_jersey_detection = executor.submit(jersey_det_model.track, source=image, conf=0.25)

        field_det_res = future_field_detection.result()[0]
        player_det_res = future_player_detection.result()[0]
        puck_det_res = future_puck_detection.result()[0]
        jersey_det_res = future_jersey_detection.result()[0]

    # ! можно сделать разные цвета для разных классов
    FIELD_FONT_SCALE = 0.5
    FIELD_COLOR = (0, 255, 0)
    
    PLAYER_FONT_SCALE = 0.3
    PLAYER_COLOR = (255, 64, 0)
    
    PUCK_FONT_SCALE = 0.3
    PUCK_COLOR = (0, 0, 255)

    # ! можно сделать разные цвета для разных классов
    JERSEY_FONT_SCALE = 0.3
    JERSEY_COLOR = (255, 255, 0)
    
    # РАЗМЕТКА BOUNDING BOX
    # # разметка поля (закоменченно потому что в конце есть строчка которая сохраняет кадр с сегментацией из YOLO)
    # for bbox in field_det_res.boxes:
        # CLASS_NAMES: dict = field_det_res.names
        # LABEL: str = f'{CLASS_NAMES[int(bbox.cls)]} {round(float(bbox.conf), 2)}'
        # POINT1: tuple = (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1]))
        # POINT2: tuple = (int(bbox.xyxy[0][2]), int(bbox.xyxy[0][3]))

        # cv2.rectangle(image, POINT1, POINT2, FIELD_COLOR)
        # cv2.putText(image, LABEL, (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, FIELD_FONT_SCALE, FIELD_COLOR, 1)

    # разметка игроков
    for bbox in player_det_res.boxes:
        CLASS_NAMES: dict = player_det_res.names
        LABEL: str = f'{CLASS_NAMES[int(bbox.cls)]} {round(float(bbox.conf), 2)}'
        POINT1: tuple = (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1]))
        POINT2: tuple = (int(bbox.xyxy[0][2]), int(bbox.xyxy[0][3]))

        cv2.rectangle(image, POINT1, POINT2, PLAYER_COLOR)
        cv2.putText(image, LABEL, (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1])-6), cv2.FONT_HERSHEY_SIMPLEX, PLAYER_FONT_SCALE, PLAYER_COLOR, 1)

    # разметка шайбы
    for bbox in puck_det_res.boxes:
        LABEL: str = f'puck {round(float(bbox.conf), 2)}'
        POINT1: tuple = (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1]))
        POINT2: tuple = (int(bbox.xyxy[0][2]), int(bbox.xyxy[0][3]))

        cv2.rectangle(image, POINT1, POINT2, PUCK_COLOR)
        cv2.putText(image, LABEL, (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1])-5), cv2.FONT_HERSHEY_SIMPLEX, PUCK_FONT_SCALE, PUCK_COLOR, 1)

    # разметка цифр на майках
    for bbox in jersey_det_res.boxes:
        CLASS_NAMES: dict = jersey_det_res.names
        LABEL: str = f'{CLASS_NAMES[int(bbox.cls)]} {round(float(bbox.conf), 2)}'
        POINT1: tuple = (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1]))
        POINT2: tuple = (int(bbox.xyxy[0][2]), int(bbox.xyxy[0][3]))
        
        cv2.rectangle(image, POINT1, POINT2, JERSEY_COLOR)
        cv2.putText(image, LABEL, (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1])-6), cv2.FONT_HERSHEY_SIMPLEX, JERSEY_FONT_SCALE, JERSEY_COLOR, 1)

    annotated_image: np.ndarray = field_det_res.plot()
    return annotated_image


def predict_video(video_path: str, output_path: str = 'output/annotated_video.mp4') -> None:
    '''Функция для разметки видео'''
    
    cap = cv2.VideoCapture(video_path)
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    FPS = cap.get(cv2.CAP_PROP_FPS)
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # проходит по всем кадрам видео, размечает их и сразу добавляет в видео
    for i in range(FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        annotated_frame = annotate_image(frame)
        out.write(annotated_frame)

    cap.release()
    out.release()


if __name__ == '__main__':
    VIDEO_PATH = '/Users/admin/Downloads/match_moment.mp4'

    predict_video(VIDEO_PATH)