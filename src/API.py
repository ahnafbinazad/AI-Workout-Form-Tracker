from flask import Flask, request, jsonify
import torch
import cv2
import time
import argparse
import torchvision.transforms as transforms
import pathlib
import os
import torch.nn.functional as F
import numpy as np

from model import build_model
from class_names import class_names as CLASS_NAMES

app = Flask(__name__)

# Define the device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform

@app.route('/infer_video', methods=['POST'])
def infer_video():
    try:
        # Get video file from request
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']

        # Save video to a temporary location
        video_path = 'temp_video.mp4'
        video_file.save(video_path)

        # Load model weights
        weights_path = pathlib.Path('../outputs/best_model.pth')
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        # Load the model
        model = build_model(
            fine_tune=False,
            num_classes=len(CLASS_NAMES)
        ).to(DEVICE).eval()
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get test transform
        transform = get_test_transform(IMAGE_RESIZE)

        cap = cv2.VideoCapture(video_path)

        workout_start_time = None
        workout_duration = 0  # in seconds
        resting_duration = 0  # in seconds
        last_classification = None

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(rgb_frame)
                input_batch = input_tensor.unsqueeze(0)

                # Move input tensor and model to the computation device
                input_batch = input_batch.to(DEVICE)
                model.to(DEVICE)

                with torch.no_grad():
                    start_time = time.time()
                    outputs = model(input_batch)
                    end_time = time.time()

                predictions = F.softmax(outputs, dim=1).cpu().numpy()
                output_class = np.argmax(predictions)
                current_output_class = np.argmax(predictions)

                if CLASS_NAMES[int(output_class)] != "resting":
                    if last_classification != "resting":
                        workout_duration += end_time - start_time
                    else:
                        workout_start_time = time.time()

                else:
                    if last_classification != "resting":
                        resting_duration += end_time - start_time

                last_classification = CLASS_NAMES[int(output_class)]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Convert durations to minutes
        workout_minutes = workout_duration / 60
        resting_minutes = resting_duration / 60

        result = {
            "workout_minutes": workout_minutes,
            "resting_minutes": resting_minutes
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
