from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from neuralNetworkSchema import generateSchema
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)
CORS(app)

# Load MNIST dataset once when server starts
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

@app.route("/process", methods=["POST"])
def process():
    f = open("test.txt", "a")
    f.write("Successfully received input from frontend\n!")
    f.close()
    file = request.files.get("file")
    hidden_layers = request.form.get("hidden_layers")
    iterations = request.form.get("iterations")
    activation_function = request.form.get("activation_function")
    training_data_count = request.form.get("training_data_count")
    image_number = request.form.get("image_number")

    if not all([iterations, hidden_layers, activation_function, training_data_count]):
        return jsonify({"error": "Missing required fields"}), 400

    if file:
        file.save(f"uploads/{file.filename}")

    response_data = {
        "message": "Received data successfully",
        "hidden_layers": hidden_layers,
        "iterations": iterations,
        "activation_function": activation_function,
        "training_data_count": training_data_count,
        "image_number": image_number,
    }

    # Only call generateSchema if image_number is not None
    if image_number and image_number != "None":
        generateSchema(image_number, iterations, hidden_layers, training_data_count, 64, activation_function, 'softmax', 10, 0.0005, True)

    return jsonify(response_data)

@app.route("/get_mnist_image/<image_number>", methods=["GET"])
def get_mnist_image(image_number):
    """Return a specific MNIST image based on its index"""
    # Check if image_number is "None" (as a string)
    if image_number == "None":
        return jsonify({"error": "No image number provided"}), 400

    try:
        # Convert to integer
        image_idx = int(image_number)

        # Ensure image_idx is valid
        if image_idx < 0 or image_idx >= len(mnist_images):
            return jsonify({"error": "Invalid image number"}), 400

        # Get the image and label
        image = mnist_images[image_idx]
        label = mnist_labels[image_idx]

        # Create a figure and save it to a BytesIO object
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        # Save the image to a BytesIO object
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.1)
        img_bytes.seek(0)
        plt.close()

        # Return the image with metadata
        return send_file(
            img_bytes,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'mnist_image_{image_idx}_label_{label}.png'
        )

    except ValueError:
        return jsonify({"error": "Image number must be an integer"}), 400

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)