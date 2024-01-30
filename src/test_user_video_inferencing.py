import unittest
import subprocess
import os
import shutil
from inference_video import infer_video


class TestVideoInference(unittest.TestCase):
    def setUp(self):
        self.test_input_path = '../inference_data/plank.mp4'
        self.test_weights_path = '../outputs/best_model.pth'
        self.expected_output_class = "plank"  # Replace with the expected output class for your test

    def test_video_inference(self):
        # Make a copy of the input video for testing
        # test_input_copy_path = '../inference_data/plank.mp4'
        # shutil.copyfile(self.test_input_path, test_input_copy_path)

        # Run the video inference
        args = ['--input', self.test_input_path, '--weights', self.test_weights_path]
        result = infer_video(args)

        # Validate the result
        self.assertEqual(result, self.expected_output_class)

        # # Clean up the copied video
        # os.remove(test_input_copy_path)


if __name__ == '__main__':
    unittest.main()
