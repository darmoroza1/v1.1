import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Шляхи до файлів
FEATURES_PATH = 'features/features.csv'
MODEL_PATH = 'model/drone_detector.pkl'

def main():
    # 1. Завантаження ознак
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=['label'])
    y = df['label']

    # 2. Розділення на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Навчання моделі
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Передбачення та оцінка
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # 5. Збереження моделі
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Модель збережено: {MODEL_PATH}")

if __name__ == '__main__':
    main()
