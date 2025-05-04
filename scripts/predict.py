import os
import sys
import librosa
import numpy as np
import joblib

# Шляхи
MODEL_PATH = 'model/drone_detector.pkl'

# Функція витягу ознак — така ж, як при навчанні
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])

    return np.hstack([mfcc, zcr, spec_centroid, rms])

def main():
    if len(sys.argv) < 2:
        print(" Будь ласка, вкажи шлях до .wav файлу як аргумент.")
        print("Приклад: python scripts/predict.py dataset/drone/drone_001.wav")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Файл не знайдено: {file_path}")
        return

    print(f"Аналіз файлу: {file_path}")

    features = extract_features(file_path).reshape(1, -1)

    model = joblib.load(MODEL_PATH)
    prediction = model.predict(features)[0]

    if prediction == 1:
        print("Результат: Дрон виявлено")
    else:
        print("Результат: Звук дрону не виявлено")

if __name__ == '__main__':
    main()
