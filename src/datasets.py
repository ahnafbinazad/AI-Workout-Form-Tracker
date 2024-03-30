import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Constants
ROOT_DIR = os.path.join('..', 'input')
IMAGE_SIZE = 256
NUM_WORKERS = 4
VALID_SPLIT = 0.10

# Transforms
def get_transform(image_size, is_training=True):
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    if is_training:
        transform_list.insert(2, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(3, transforms.RandomRotation(35))
    return transforms.Compose(transform_list)

def prepare_datasets():
    dataset = datasets.ImageFolder(
        ROOT_DIR,
        transform=get_transform(IMAGE_SIZE)
    )
    dataset_size = len(dataset)
    valid_size = int(VALID_SPLIT * dataset_size)
    train_set, valid_set = random_split(dataset, [dataset_size - valid_size, valid_size])
    return train_set, valid_set, dataset.classes

def prepare_data_loaders(train_set, valid_set, batch_size):
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader
