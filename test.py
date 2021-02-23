import argparse
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_model, get_data_loader, calculate_metrics
from config import NORMALIZE_MEAN, NORMALIZE_STD, CLASS_NAMES


def test(weights_path: Optional[str], only_show: bool) -> None:
    """
    Testing classifier and visualizing predictions.

    :param weights_path: path to saved weights or None.
    :param only_show: if True then only show predictions and labels. If False, calculate test metric before showing.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader = get_data_loader(is_train=False)
    class_names = test_loader.dataset.classes
    class_names = [CLASS_NAMES[class_names[i]] for i in range(len(class_names))]
    testing_samples_num = len(test_loader.dataset)

    model = get_model(weights_path)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    # Set model to evaluating mode.
    model.eval()
    running_loss, running_corrects = 0.0, 0

    if not only_show:
        all_labels, all_predictions = [], []
        for inputs, labels in tqdm(test_loader, desc='Testing. Batch'):
            all_labels.append(labels.numpy())
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_func(outputs, labels)
            all_predictions.append(preds.cpu().numpy())

            # Get loss and corrects.
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / testing_samples_num
        epoch_acc = running_corrects.double() / testing_samples_num
        print('Testing: loss = {:.4f}, accuracy = {:.4f}.\n'.format(epoch_loss, epoch_acc))
        report = calculate_metrics(all_labels, all_predictions, class_names)
        print(report)

    test_loader = get_data_loader(is_train=False, batch_size=1)
    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            outputs_softmax = torch.nn.Softmax(-1).forward(outputs)

        image = inputs.cpu().numpy()[0, :, :, :].transpose((1, 2, 0))
        image = np.clip(image * NORMALIZE_STD + NORMALIZE_MEAN, 0, 1)
        label_class = class_names[labels.numpy()[0]]
        pred_class_index = preds.cpu().numpy()[0]
        pred_class = class_names[pred_class_index]
        pred_value = outputs_softmax.cpu().numpy()[0, pred_class_index]

        plt.imshow(image)
        plt.title('Prediction: {}, {:.02f}%.\nLabel: {}.'.format(pred_class, pred_value * 100, label_class))
        if plt.waitforbuttonpress(0):
            plt.close('all')
            return


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Script for evaluating imagewoof classifier and visualizing predictions')
    parser.add_argument('--show', action='store_true', help='Only show predictions without metrics evaluating.')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generator\'s seed. If seed < 0, seed will be None.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    if args.weights is None:
        print('WARNING: using randomly initialized model for evaluating.')

    test(args.weights, args.show)
