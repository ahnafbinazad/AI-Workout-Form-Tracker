from flask import Flask, request

app = Flask(__name__)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return {'error': 'No video file provided'}, 400

    video = request.files['video']
    # Process the video (e.g., save it to a folder, perform inference, etc.)
    # Add your logic here based on your requirements.

    return {'message': 'Video uploaded successfully'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
