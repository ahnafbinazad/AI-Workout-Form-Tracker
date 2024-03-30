import torch
import numpy as np
import cv2
import os
import torchvision.transforms as transforms
import glob
import pathlib

from torch.testing._internal.common_utils import args

from model import build_model
from class_names import class_names as CLASS_NAMES

# Constants
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256

# Transforms
def get_test_transform(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def annotate_image(image, output_class):
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

def inference(model, image, DEVICE):
    model.eval()
    with torch.no_grad():
        image = image.to(DEVICE)
        outputs = model(image)
    predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    result = annotate_image(image, predictions)
    return result

if __name__ == '__main__':
    weights_path = pathlib.Path(args.weights)
    infer_result_path = os.path.join(
        '..', 'outputs', 'inference_results', 'image_outputs'
    )
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_image_paths = glob.glob(os.path.join(args.input, '*.jpg'))
    transform = get_test_transform(IMAGE_RESIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i+1}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        result = inference(model, image, DEVICE)
        image_name = image_path.split(os.path.sep)[-1]
        cv2.imshow('Image', result)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(infer_result_path, image_name), result*255.)
