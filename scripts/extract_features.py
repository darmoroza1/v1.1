import os
import librosa
import numpy as np
import pandas as pd

# Шляхи до файлів
METADATA_PATH = 'dataset/metadata.csv'
AUDIO_BASE_PATH = 'dataset/'
FEATURES_OUTPUT_PATH = 'features/features.csv'

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # MFCC (13 коефіцієнтів)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])

    return np.hstack([mfcc, zcr, spec_centroid, rms])

def main():
    metadata = pd.read_csv(METADATA_PATH)
    features = []
    labels = []

    for _, row in metadata.iterrows():
        file_path = os.path.join(AUDIO_BASE_PATH, row['filename'])
        try:
            feats = extract_features(file_path)
            features.append(feats)
            labels.append(row['label'])
        except Exception as e:
            print(f"Помилка з {file_path}: {e}")

    # Формування датафрейму
    columns = [f'mfcc_{i}' for i in range(13)] + ['zcr', 'spectral_centroid', 'rms']
    df = pd.DataFrame(features, columns=columns)
    df['label'] = labels

    os.makedirs(os.path.dirname(FEATURES_OUTPUT_PATH), exist_ok=True)
    df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    print(f"Збережено: {FEATURES_OUTPUT_PATH}")

if __name__ == '__main__':
    main()
