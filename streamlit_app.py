import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile

# Завантаження моделі
MODEL_PATH = 'model/drone_detector.pkl'
model = joblib.load(MODEL_PATH)

# Витяг ознак
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])
    return np.hstack([mfcc, zcr, centroid, rms])

# Інтерфейс
st.title("Виявлення звуку дрону")
st.write("Завантаж `.wav` файл, щоб перевірити, чи містить він звук дрону.")

uploaded_file = st.file_uploader("Оберіть аудіофайл", type=["wav"])

if uploaded_file is not None:
    # Зберегти тимчасово
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Аналіз
    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("Дрон виявлено")
        else:
            st.info("Звук дрону не виявлен")
    except Exception as e:
        st.error(f"Помилка обробки: {e}")
