import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "models/depth_anything_v3.tflite"
IMAGE_PATH = "test_images/test.jpg"
OUTPUT_PATH = "output/depth.png"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((518, 518))

x = np.array(img).astype(np.float32) / 255.0
x = np.expand_dims(x, axis=0)

import time

interpreter.set_tensor(input_details[0]["index"], x)

start = time.time()
interpreter.invoke()
end = time.time()

print(f"Inference time: {(end - start) * 1000:.2f} ms")

depth = interpreter.get_tensor(output_details[0]["index"])[0, :, :, 0]

depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

plt.imsave(OUTPUT_PATH, depth_norm, cmap="inferno")

print("Input:", input_details)
print("Output:", output_details)
print("Saved depth map to", OUTPUT_PATH)