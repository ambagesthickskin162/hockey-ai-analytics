from ultralytics import YOLO
import os
import yaml
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path
from utils.base_config import BaseConfig
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))


def update_data_yaml(config: BaseConfig):
    """
    Функция для обновления путей в data.yaml на абсолютные.
    
    :param data_yaml_path: Путь к файлу data.yaml.
    """

    dir_path = Path(__file__).resolve().parents[2]
    dataset_path = os.path.join(dir_path, 'data', config.dataset_path)
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')

    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    if config.task_name == 'player_detection' and not os.path.exists(os.path.join(dataset_path, 'train')):
        for stage in ['train', 'valid']:
            for type_content in ['images', 'labels']:
                os.makedirs(os.path.join(dataset_path, stage, type_content), exist_ok=True)
                for path in os.listdir(os.path.join(dataset_path, type_content, stage)):
                    shutil.copy(os.path.join(dataset_path, type_content, stage, path),
                                os.path.join(dataset_path, stage, type_content, path))
        shutil.rmtree(os.path.join(dataset_path, 'images'), ignore_errors=True)
        shutil.rmtree(os.path.join(dataset_path, 'labels'), ignore_errors=True)

    # Обновление путей в data.yaml
    data['train'] = os.path.abspath(os.path.join(dir_path, 'data', config.dataset_path, 'train/images'))
    data['val'] = os.path.abspath(os.path.join(dir_path, 'data', config.dataset_path, 'valid/images'))
    if 'test' in data:
        data['test'] = os.path.abspath(os.path.join(dir_path, 'data', config.dataset_path, 'test/images'))
    
    for key in list(data.keys()):
        if key not in ['train', 'val', 'test', 'names']:
            print(f'Из data.yaml удален ключ "{key}"')
            del data[key]

    # Сохранение изменений
    with open(data_yaml_path, 'w') as file:
        yaml.safe_dump(data, file)

    print(f"Updated paths in {data_yaml_path} to absolute paths.")


def train_yolo(config: BaseConfig):
    """
    Функция для тренировки YOLO модели с использованием библиотеки ultralytics.
    
    :param data_yaml_path: Путь к data.yaml файлу.
    :param config_path: Путь к файлу конфигурации YOLO (необязательный, для кастомных конфигураций).
    :param weights_path: Путь к файлу начальных весов (по умолчанию используется yolov8n.pt).
    :param output_dir: Папка для сохранения результатов тренировки.
    :param epochs: Количество эпох для обучения (по умолчанию 100).
    :param batch_size: Размер батча (по умолчанию 16).
    :param img_size: Размер входного изображения (по умолчанию 640).
    """
    # Загрузка модели, либо с кастомной конфигурацией, либо с предустановленными весами
    if config.model is not None:
        model = YOLO(config.model)  # загрузка предобученной модели с весами
    else:
        model = YOLO('yolov8n.pt')  # по умолчанию используется YOLOv8 Nano модель

    dir_path = Path(__file__).resolve().parents[2]
    data_yaml_path = os.path.join(dir_path, 'data', config.dataset_path, 'data.yaml')
    output_dir = os.path.join(dir_path, 'runs/train')

    if os.path.exists(output_dir):
        for path in os.listdir(output_dir):
            if config.task_name in path:
                shutil.rmtree(os.path.join(output_dir, path))
                print(f'Удалена папка с результатами: {os.path.join(output_dir, path)}')

    model.train(
        data=data_yaml_path,
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.image_size,
        project=output_dir,
        name=config.task_name,


        workers=10,
        degrees=3.0,
        perspective	= 0.0001,
        mixup=0.02,
        shear=3.0
    )

    # Получение пути к лучшим весам
    output_weights_path = os.path.join(output_dir, config.task_name, 'weights', 'best.pt')
    best_weights_path = os.path.join(dir_path, "weights", f"{config.task_name}.pt")

    # Копирование лучших весов в целевую папку

    os.makedirs(os.path.join(dir_path, 'weights'), exist_ok=True)
    shutil.copy(output_weights_path, best_weights_path)

    print(f"Training completed. Results saved to {output_dir}.")


def check_correct_yaml(dataset_dir: str, task_config: BaseConfig):
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(data_yaml_path):
        return False
    
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    if task_config.task_name in ['field_detection'] and \
        (data['roboflow']['version'] != task_config.dataset_version or data['roboflow']['project'] != task_config.dataset_project):
        return False

    return True
    

def check_and_download_dataset(task_config: BaseConfig):
    """
    Функция для проверки наличия датасета и загрузки его с Roboflow при необходимости.

    """
    # Проверка наличия датасета

    # dataset_roboflow_id = task_config.dataset_roboflow_id

    dir_path = Path(__file__).resolve().parents[2]
    dataset_dir = os.path.join(dir_path, 'data', task_config.dataset_path)

    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 1 and check_correct_yaml(dataset_dir, task_config):
        print(f"Dataset already exists at {dataset_dir}.")
    else:
        project = rf.workspace(task_config.dataset_workspace).project(task_config.dataset_project)
        version = project.version(task_config.dataset_version)
        dataset = version.download("yolov8", location=dataset_dir, overwrite=True)
        print(dataset)
