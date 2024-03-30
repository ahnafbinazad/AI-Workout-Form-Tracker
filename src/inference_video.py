import torch
import cv2
import time
import argparse
import torchvision.transforms as transforms
import pathlib
import os
import numpy as np

from model import build_model
from class_names import class_names as CLASS_NAMES

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/video_1.mp4', help='path to the input video')
parser.add_argument('-w', '--weights', default='../outputs/best_model.pth', help='path to the model weights')
args = parser.parse_args()

# Output directory
OUT_DIR = '../outputs/inference_results/video_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256


# Validation transforms
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


if __name__ == '__main__':
    # Load the model
    weights_path = pathlib.Path(args.weights)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE).eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = get_test_transform(IMAGE_RESIZE)

    # Video capture
    cap = cv2.VideoCapture(args.input)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = args.input.split('/')[-1].split('.')[0]
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (frame_width, frame_height))
    frame_count = 0
    total_fps = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_frame)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_batch)
                end_time = time.time()

            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            output_class = np.argmax(predictions)

            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"{CLASS_NAMES[int(output_class)]}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)
            cv2.imshow('Result', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
