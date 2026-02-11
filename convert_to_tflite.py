import tensorflow as tf
import os

MODEL_PATH = "model.h5"
TFLITE_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: {MODEL_PATH} not found.")
    exit(1)

print(f"🔄 Loading {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

print("⚙️ Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print(f"💾 Saving to {TFLITE_PATH}...")
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete!")
print(f"Original Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
print(f"TFLite Size:   {os.path.getsize(TFLITE_PATH) / (1024*1024):.2f} MB")
