import numpy as np
import os
import io
import base64

# ✅ Fix for non-GUI environments (Flask, servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2DTranspose, LeakyReLU


latent_dim = 128  # change this to your actual latent dimension

# Rebuild the decoder architecture
def build_decoder(latent_dim):
    decoder = Sequential()
    decoder.add(Dense(1024, activation='selu', input_shape=(latent_dim,)))
    decoder.add(BatchNormalization())
    decoder.add(Dense(8192, activation='selu'))
    decoder.add(Reshape((4, 4, 512)))
    
    filters_decode = [256, 128, 64, 32]
    for i in filters_decode:
        decoder.add(Conv2DTranspose(i, (5,5), activation=LeakyReLU(0.02), strides=2, padding='same'))
        decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(3, (5,5), activation='sigmoid', strides=1, padding='same'))
    decoder.add(BatchNormalization())
    
    return decoder

# Load decoder and weights
decoder = build_decoder(latent_dim)
decoder.load_weights('decoder_weights.weights.h5')  # path to your saved weights file

def generate_image_base64():
    latent_vector = np.random.uniform(-1, 1, (1, latent_dim))
    prediction = decoder.predict(latent_vector)
    
    image = prediction[0] * 255
    image = image.astype(np.uint8)

    # Save to a buffer in memory
    buf = io.BytesIO()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

# Example usage:
if __name__ == "__main__":
    print("Testing image generation...")
    b64_img = generate_image_base64()
    
    # Optional: save base64 image to verify visually
    output_path = "test_output.jpg"
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_img))
    
    print(f"Test image saved as: {output_path}")

# print first 100 chars of base64 string for sanity check
