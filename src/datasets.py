import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Constants
ROOT_DIR = os.path.join('..', 'input')  # Directory containing the dataset
IMAGE_SIZE = 256  # Image size after resizing
NUM_WORKERS = 4  # Number of parallel processes for data loading
VALID_SPLIT = 0.10  # Ratio of data to be used for validation


# Training transforms
def get_train_transform(image_size):
    """Returns a composition of image transformations for training."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the image
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        transforms.RandomRotation(35),  # Randomly rotate the image
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )  # Normalize the image using predefined mean and standard deviation
    ])
    return train_transform


# Validation transforms
def get_valid_transform(image_size):
    """Returns a composition of image transformations for validation."""
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the image
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )  # Normalize the image using predefined mean and standard deviation
    ])
    return valid_transform


def get_datasets():
    """
    Prepares the Datasets.
    Returns the training and validation datasets along
    with the class names.
    """
    # Initialize datasets with respective transforms
    dataset = datasets.ImageFolder(ROOT_DIR, transform=(get_train_transform(IMAGE_SIZE)))
    dataset_test = datasets.ImageFolder(ROOT_DIR, transform=(get_valid_transform(IMAGE_SIZE)))
    dataset_size = len(dataset)

    # Calculate the validation dataset size
    valid_size = int(VALID_SPLIT * dataset_size)

    # Randomize the data indices
    indices = torch.randperm(len(dataset)).tolist()

    # Training and validation sets
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])

    return dataset_train, dataset_valid, dataset.classes


def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    # DataLoader for training set with shuffling
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS
    )

    # DataLoader for validation set without shuffling
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, valid_loader
