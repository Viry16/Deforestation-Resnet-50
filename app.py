import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

# =====================
# 1. Load Model
# =====================
MODEL_PATH = r"D:\PU\Semester 4\Python\Deep Learning\Final Project\resnet50_deforestation.keras"

# Optional: Check if file exists first
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


# =====================
# 2. Load Label Mapping
# =====================
# Biasanya dari train_generator.class_indices
# Contoh default (ubah sesuai dataset kamu)
class_indices = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9
}
# urutkan sesuai index
class_labels = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

# =====================
# 3. Streamlit UI
# =====================
st.title("🌍 Deforestation Classification App")
st.write("Upload citra satelit untuk memprediksi jenis lahan.")

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # tampilkan gambar
    st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

    # =====================
    # 4. Preprocessing
    # =====================
    img = image.load_img(uploaded_file, target_size=(224, 224))  # karena ResNet50
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # =====================
    # 5. Prediksi
    # =====================
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = class_labels[pred_idx]
    pred_prob = preds[0][pred_idx]

    # =====================
    # 6. Output
    # =====================
    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Kelas Prediksi:** {pred_label}")
    st.write(f"**Probabilitas:** {pred_prob:.4f}")

    # tampilkan semua probabilitas dengan label
    import pandas as pd
    df_probs = pd.DataFrame({
        "Probabilitas": preds[0]
    }, index=class_labels)

    st.bar_chart(df_probs)  # bar chart pakai label
    st.dataframe(df_probs.style.format({"Probabilitas": "{:.4f}"}))  # tabel rapi