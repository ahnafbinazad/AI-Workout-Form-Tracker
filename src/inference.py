import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib

from model import build_model
from class_names import class_names as CLASS_NAMES

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', 
    default='../outputs/best_model.pth',
    help='path to the model weights',
)
parser.add_argument(
    '--input',
    required=True,
    help='path to the directory containing images'
)
args = parser.parse_args()

# Constants and other configurations.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
IMAGE_RESIZE = 256  # Image size after resizing

# Validation transforms
def get_test_transform(image_size):
    """Returns a composition of image transformations for testing."""
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),  # Resize the image
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )  # Normalize the image using predefined mean and standard deviation
    ])
    return test_transform

def denormalize(
    x,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    """Denormalizes a tensor."""
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def annotate_image(image, output_class):
    """Annotates the image with the predicted class."""
    image = denormalize(image).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_name = CLASS_NAMES[int(output_class)]
    cv2.putText(
        image,
        class_name,
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA
    )
    return image

def inference(model, testloader, DEVICE):
    """
    Function to run inference.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    """
    model.eval()
    counter = 0
    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(DEVICE)

        # Forward pass.
        outputs = model(image)
    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    # Predicted class number.
    output_class = np.argmax(predictions)
    # Show and save the results.
    result = annotate_image(image, output_class)
    return result

if __name__ == '__main__':
    weights_path = pathlib.Path(args.weights)
    infer_result_path = os.path.join(
        '..', 'outputs', 'inference_results', 'image_outputs'
    )
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    # Load the model.
    model = build_model(
        fine_tune=False,
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_image_paths = glob.glob(os.path.join(args.input, '*.jpg'))

    transform = get_test_transform(IMAGE_RESIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i+1}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        result = inference(
            model,
            image,
            DEVICE
        )
        # Save the image to disk.
        image_name = image_path.split(os.path.sep)[-1]
        cv2.imshow('Image', result)
        cv2.waitKey(1)
        cv2.imwrite(
            os.path.join(infer_result_path, image_name), result*255.
        )
