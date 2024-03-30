import torch
import argparse
import os
import numpy as np

from tqdm.auto import tqdm
from model import build_model
from datasets import prepare_data_loaders, prepare_datasets
from utils import save_model, save_plots, SaveBestModel

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train the network')
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('-ft', '--fine-tune', action='store_true', help='Fine-tune all layers')
parser.add_argument('--save-name', default='model', help='File name of the final model to save')
parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
args = parser.parse_args()

# Training function
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    prog_bar = tqdm(trainloader, total=len(trainloader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for i, data in enumerate(prog_bar):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function
def validate(model, testloader, criterion, device):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    prog_bar = tqdm(testloader, total=len(testloader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    with torch.no_grad():
        for i, data in enumerate(prog_bar):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    out_dir = os.path.join('..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    dataset_train, dataset_valid, class_names = prepare_datasets()
    train_loader, valid_loader = prepare_data_loaders(dataset_train, dataset_valid, batch_size=args.batch_size)

    lr = args.learning_rate
    epochs = args.epochs
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(fine_tune=args.fine_tune, num_classes=len(class_names)).to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    save_best_model = SaveBestModel()

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1, verbose=True) if args.scheduler else None

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        save_best_model(valid_epoch_loss, epoch, model, out_dir, args.save_name)
        if scheduler:
            scheduler.step()
        print('-'*50)

    save_model(epochs, model, optimizer, criterion, out_dir, args.save_name)
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')
