from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS
from neuralNetworkSchema import generateSchema
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os
import traceback
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app)

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://neural-network-visualizer-sir5.onrender.com",
            "https://neuralnetworkvisualizer.co",
            "https://www.neuralnetworkvisualizer.co",
            "http://localhost:3000",
            "http://localhost:5000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

checkpoints_dir = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def run_generate_schema(image_number, iterations, hidden_layers, training_data_count, activation_function):
    try:
        plt.close('all')
        plt.ioff()
        generateSchema(
            int(image_number),
            int(iterations),
            int(hidden_layers),
            int(training_data_count),
            64,
            str(activation_function),
            'softmax',
            10,
            0.0005,
            True
        )
    except Exception as e:
        print(f"Error in schema generation: {e}")
    finally:
        plt.close('all')

@app.route("/get_mnist_image/<image_number>", methods=["GET"])
def get_mnist_image(image_number):
    print(f"Received MNIST image request for number: {image_number}")
    print(f"Total MNIST images available: {len(mnist_images)}")

    if image_number == "None":
        print("No image number provided")
        return jsonify({"error": "No image number provided"}), 400

    try:
        image_idx = int(image_number)

        if image_idx < 0 or image_idx >= len(mnist_images):
            print(f"Invalid image number: {image_idx}")
            return jsonify({
                "error": "Invalid image number",
                "details": f"Number must be between 0 and {len(mnist_images)-1}"
            }), 400

        image = mnist_images[image_idx]
        label = mnist_labels[image_idx]

        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray')
        plt.title(f'MNIST Digit: {label}')
        plt.axis('off')

        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
        img_bytes.seek(0)
        plt.close()

        return send_file(
            img_bytes,
            mimetype='image/png',
            as_attachment=False,
            download_name=f'mnist_image_{image_idx}_label_{label}.png'
        )

    except ValueError:
        print(f"Invalid image number format: {image_number}")
        return jsonify({
            "error": "Invalid image number format",
            "details": "Image number must be an integer"
        }), 400
    except Exception as e:
        print(f"Unexpected error in get_mnist_image: {e}")
        return jsonify({
            "error": "Unexpected error",
            "details": str(e)
        }), 500

@app.route("/process", methods=["POST"], cors=False)
def process():
    print("Process Route - Request Method:", request.method)
    print("Process Route - Request Headers:", request.headers)
    print("Process Route - Request Origin:", request.headers.get('Origin'))

    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "https://neuralnetworkvisualizer.co")
    response.headers.add("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")

    file = request.files.get("file")
    hidden_layers = request.form.get("hidden_layers")
    iterations = request.form.get("iterations")
    activation_function = request.form.get("activation_function")
    training_data_count = request.form.get("training_data_count")
    image_number = request.form.get("image_number")
    print("Process route entered")
    video_path = "checkpoints/neural_network_animation.mp4"
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        print(f"Creating checkpoints directory: {checkpoints_dir}")
        os.makedirs(checkpoints_dir)
    print(f"Current working directory: {os.getcwd()}")
    abs_video_path = os.path.abspath(video_path)
    print(f"Absolute video path: {abs_video_path}")
    if image_number and image_number != "None":
        try:
            generateSchema(
                int(image_number),
                int(iterations),
                int(hidden_layers),
                int(training_data_count),
                64,
                str(activation_function),
                'softmax',
                10,
                0.0005,
                True
            )
            print("Checking for video file after generation:")
            print(f"Video path exists: {os.path.exists(video_path)}")
            if os.path.exists(checkpoints_dir):
                print("Checkpoints directory contents:")
                for filename in os.listdir(checkpoints_dir):
                    print(f"- {filename}")
            time.sleep(2)
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"Video file size: {file_size} bytes")
                if file_size > 0:
                    return send_file(video_path, mimetype='video/mp4')
            return jsonify({
                "error": "Video generation failed",
                "details": "No video file found after generation",
                "video_path": abs_video_path
            }), 404
        except Exception as e:
            print(f"Exception during schema generation: {e}")
            traceback.print_exc()
            return jsonify({
                "error": "Schema generation failed",
                "details": str(e)
            }), 500
    return jsonify({"error": "No image number provided"}), 400

if __name__ == "__main__":
    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
