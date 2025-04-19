import io
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, Response, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)
generator = tf.keras.models.load_model("D:\AI_Projects\GEN-NY\GEN-NY\models\generator_model.h5")  # Update with your model path


# Function to generate an AI image
def generate_image(seed_value):
    noise_dim = 100
    np.random.seed(seed_value)
    random_noise = np.random.randn(1, noise_dim)
    generated_image = generator.predict(random_noise)

    # Normalize and convert to uint8
    generated_image = (generated_image + 1) / 2
    generated_image_uint8 = (generated_image[0] * 255).astype(np.uint8)

    # Upscale image for better resolution
    upscaled_image = cv2.resize(generated_image_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)

    # Convert to PIL Image and stream it directly
    img = Image.fromarray(upscaled_image)
    img_io = io.BytesIO()
    img.save(img_io, format="PNG")
    img_io.seek(0)

    return img_io


@app.route('/')
def home():
    return "<h1>Welcome to the GAN Image Generator</h1><p>Use the frontend to generate images.</p>"


@app.route('/generate', methods=['GET'])
def generate():
    seed_value = int(request.args.get("seed", np.random.randint(10000)))
    print(f"Generating image for seed: {seed_value}")

    img_io = generate_image(seed_value)
    print("Sending image response...")

    response = Response(img_io.getvalue(), mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"

    return response


if __name__ == "__main__":
    app.run(debug=True)
