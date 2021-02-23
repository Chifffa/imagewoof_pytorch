import datetime
from typing import Optional, List

import cv2
import torch
import torchvision
import numpy as np
from sklearn.metrics import classification_report
from efficientnet_pytorch import EfficientNet

from config import (
    MODEL_TYPE, NUM_CLASSES, TRAIN_PATH, TEST_PATH, TRAIN_DATA_TRANSFORMS, TEST_DATA_TRANSFORMS, BATCH_SIZE, NUM_WORKERS
)


def get_date() -> str:
    """
    Create string with current date and time.

    :return: created string.
    """
    t = datetime.datetime.now().timetuple()
    return '___{:02d}.{:02d}.{:02d}_{:02d}_{:02d}_{:02d}'.format(*[t[x] for x in [2, 1, 0, 3, 4, 5]])


def get_model(weights_path: Optional[str] = None) -> torch.nn.Module:
    """
    Getting model with provided weights.

    :param weights_path: path to saved weights or None.
    :return: EfficientNet model with provided weights.
    """
    model = EfficientNet.from_name(MODEL_TYPE, num_classes=NUM_CLASSES)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def get_data_loader(is_train: bool = True, batch_size: int = BATCH_SIZE) -> torch.utils.data.DataLoader:
    """
    Getting training or testing data loader.

    :param is_train: True to get trainig data loader or False to get testing one.
    :param batch_size: batch size.
    :return: prepared DataLoader object.
    """
    if is_train:
        root_path = TRAIN_PATH
        transforms = TRAIN_DATA_TRANSFORMS
    else:
        root_path = TEST_PATH
        transforms = TEST_DATA_TRANSFORMS

    dataset_folder = torchvision.datasets.DatasetFolder(
        root=root_path,
        loader=lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
        transform=transforms,
        is_valid_file=lambda x: True
    )
    # Shuffle only once testing data.
    if not is_train:
        np.random.shuffle(dataset_folder.samples)
    data_loader = torch.utils.data.DataLoader(
        dataset_folder, batch_size=batch_size, shuffle=is_train, num_workers=NUM_WORKERS
    )
    return data_loader


def calculate_metrics(labels: List[np.ndarray], predicts: List[np.ndarray],
                      label_names: Optional[List[str]] = None) -> str:
    """
    Counting metrics using sklearn classification report.

    :param labels: list with labels numpy arrays.
    :param predicts: list with predictions numpy arrays.
    :param label_names: list with class names.
    :return: classification report as string.
    """
    all_labels = np.concatenate(labels, axis=0)
    all_predicts = np.concatenate(predicts, axis=0)
    report = classification_report(
        all_labels,
        all_predicts,
        target_names=label_names
    )
    return report
