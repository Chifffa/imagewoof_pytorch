import os
from multiprocessing import cpu_count

from torchvision import transforms

# Dataset parameters.
DATASET_PATH = 'imagewoof2'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'val')

# Path to saved weights.
WEIGHTS_PATH = 'weights'

# Training parameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_WORKERS = min(cpu_count(), 10)

CLASS_NAMES = {
    'n02099601': 'Golden retriever',
    'n02086240': 'Shih-Tzu',
    'n02087394': 'Rhodesian ridgeback',
    'n02105641': 'Old English sheepdog',
    'n02088364': 'Beagle',
    'n02111889': 'Samoyed',
    'n02093754': 'Border terrier',
    'n02089973': 'English foxhound',
    'n02096294': 'Australian terrier',
    'n02115641': 'Dingo'
}
MODEL_TYPE = 'efficientnet-b0'
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (224, 224, 3)

NORMALIZE_MEAN = [0, 0, 0]
NORMALIZE_STD = [255, 255, 255]
TRAIN_DATA_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(INPUT_SHAPE[:2]),
    transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=5, scale=(0.8, 1.2), fillcolor=None),
            transforms.GaussianBlur(kernel_size=3)
    ]),
    transforms.Normalize([0, 0, 0], [255, 255, 255])
])
TEST_DATA_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(INPUT_SHAPE[:2]),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])
