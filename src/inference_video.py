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


def infer_video(args):
    OUT_DIR = '../outputs/inference_results/video_outputs'
    os.makedirs(OUT_DIR, exist_ok=True)

    # Set the computation device.
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    IMAGE_RESIZE = 256

    # Validation transforms.
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

    weights_path = pathlib.Path(args.weights)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    # Load the model.
    model = build_model(
        fine_tune=False,
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE).eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = get_test_transform(IMAGE_RESIZE)

    cap = cv2.VideoCapture(args.input)
    # Get the frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # file saving section
    # # Define the outfile file name.
    # save_name = f"{args.input.split('/')[-1].split('.')[0]}"
    # # Define codec and create VideoWriter object.
    # out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4",
    #                       cv2.VideoWriter_fourcc(*'mp4v'), 30,
    #                       (frame_width, frame_height))

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0

    # Initialize variables to keep track of the highest probability and its corresponding class
    output_class_counts = {}

    while cap.isOpened():
        # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transforms to the input image.
            input_tensor = transform(rgb_frame)
            # Set the batch dimension.
            input_batch = input_tensor.unsqueeze(0)

            # Move the input tensor and model to the computation device.
            input_batch = input_batch.to(DEVICE)
            model.to(DEVICE)

            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_batch)
                end_time = time.time()

            # Get the softmax probabilities.
            predictions = F.softmax(outputs, dim=1).cpu().numpy()
            # Get the top 1 prediction.
            output_class = np.argmax(predictions)
            current_output_class = np.argmax(predictions)

            # Update the dictionary with the current output class
            output_class_counts[current_output_class] = output_class_counts.get(current_output_class, 0) + 1

            # Get the current fps.
            fps = 1 / (end_time - start_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            # Section where the result is showed in video
            # cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2)
            # cv2.putText(
            #     frame,
            #     f"{CLASS_NAMES[int(output_class)]}",
            #     (15, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 255, 0),
            #     2
            # )
            # cv2.imshow('Result', frame)

            # this section saves the result file
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    # Print the output class and its count
    for output_class, count in output_class_counts.items():
        print(f"Output Class: {CLASS_NAMES[int(output_class)]}, Count: {count}")

    # Find the output class with the highest count
    most_frequent_output_class = max(output_class_counts, key=output_class_counts.get)

    # Print the output class with the highest count
    print(
        f"Most Frequent Output Class: {CLASS_NAMES[int(most_frequent_output_class)]}, Count: {output_class_counts[most_frequent_output_class]}")

    return most_frequent_output_class


if __name__ == '__main__':
    # Construct the argument parser to parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        default='../inference_data/barbell_bicep_curl.mp4',
        help='path to the input video'
    )
    parser.add_argument(
        '-w', '--weights',
        default='../outputs/best_model.pth',
        help='path to the model weights',
    )
    args = parser.parse_args()

    result = infer_video(args)
    print(f"Final Output Class: {CLASS_NAMES[int(result)]}")
