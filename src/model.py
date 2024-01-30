from torchvision import models
import torch.nn as nn


def build_model(fine_tune=True, num_classes=10):
    # Load a pre-trained ResNet-50 model from torchvision.
    model = models.resnet50(weights='DEFAULT')

    # Fine-tune or freeze layers based on the 'fine_tune' parameter.
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Modify the final fully connected layer to match the number of classes.
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    return model


if __name__ == '__main__':
    # Build the model with default settings.
    model = build_model()

    # Print the model architecture.
    print(model)

    # Calculate and print the total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
