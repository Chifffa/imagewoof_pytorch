import os
import copy
import time
import argparse
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_model, get_data_loader, get_date
from config import NORMALIZE_MEAN, NORMALIZE_STD, CLASS_NAMES, WEIGHTS_PATH, LEARNING_RATE, EPOCHS


def show_training_images() -> None:
    """
    Showing augmented training images with labels. Click to continue or press any key to exit.
    """
    print(show_training_images.__doc__)
    train_loader = get_data_loader(is_train=True)
    classes = train_loader.dataset.classes
    for inputs, labels in train_loader:
        inputs = inputs.numpy().transpose((0, 2, 3, 1))
        labels = labels.numpy()
        for i in range(inputs.shape[0]):
            image = np.clip(inputs[i, :, :, :] * NORMALIZE_STD + NORMALIZE_MEAN, 0, 1)
            plt.imshow(image)
            plt.title(CLASS_NAMES[classes[labels[i]]])
            if plt.waitforbuttonpress(0):
                plt.close('all')
                return


def train(weights_path: Optional[str], save_path: str) -> None:
    """
    Training classifier.

    :param weights_path: path to saved weights or None.
    :param save_path: path to save weights.
    """
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = get_data_loader(is_train=True)
    test_loader = get_data_loader(is_train=False)
    training_samples_num = len(train_loader.dataset)
    testing_samples_num = len(test_loader.dataset)

    model = get_model(weights_path)
    model = model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('Starting training.')
    start_time = time.time()
    best_test_acc = 0.0
    best_epoch_num = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    try:
        for epoch in range(1, EPOCHS + 1):
            print()
            # Set model to training mode.
            model.train()
            running_loss, running_corrects = 0.0, 0

            # Iterating over training data.
            for inputs, labels in tqdm(train_loader, desc='Training. Epoch {}/{}. Batch'.format(epoch, EPOCHS)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients.
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # Forward.
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_func(outputs, labels)
                    # Backward.
                    loss.backward()
                    optimizer.step()

                # Get loss and corrects.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / training_samples_num
            epoch_acc = running_corrects.double() / training_samples_num
            print('Training: loss = {:.4f}, accuracy = {:.4f}.'.format(epoch_loss, epoch_acc))

            # Set model to evaluating mode.
            model.eval()
            running_loss, running_corrects = 0.0, 0

            # Iterating over testing data.
            for inputs, labels in tqdm(test_loader, desc='Testing. Epoch {}/{}. Batch'.format(epoch, EPOCHS)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients.
                optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    # Forward.
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_func(outputs, labels)

                # Get loss and corrects.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / testing_samples_num
            epoch_acc = running_corrects.double() / testing_samples_num
            print('Testing: loss = {:.4f}, accuracy = {:.4f}.'.format(epoch_loss, epoch_acc))

            if epoch_acc > best_test_acc:
                best_test_acc = epoch_acc
                best_epoch_num = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            # Saving weights after every epoch.
            torch.save(model.state_dict(), os.path.join(save_path, '{:03d}_epoch.pts'.format(epoch)))
    except KeyboardInterrupt:
        print('\nStopped training.')
    finally:
        finish_time = time.time() - start_time
        print('\nTraining complete in {:.0f}m {:.0f}s.'.format(finish_time // 60, finish_time % 60))
        print('Best testing accuracy: {:4f} (epoch {}).'.format(best_test_acc, best_epoch_num))
        torch.save(best_model_wts, os.path.join(save_path, 'best.pts'))


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('Script for training imagewoof classifier.')
    parser.add_argument('--show', action='store_true', help='Only show augmented images without training.')
    parser.add_argument('-sp', '--save_path', type=str, default=WEIGHTS_PATH + get_date(),
                        help='Directory to save weights.')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generator\'s seed. If seed < 0, seed will be None.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    if args.show:
        show_training_images()
    else:
        train(args.weights, args.save_path)
