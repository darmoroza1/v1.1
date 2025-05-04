import os
import pandas as pd

data = []

# Звуки дронів
for fname in os.listdir('dataset/drone'):
    if fname.endswith('.wav'):
        data.append({'filename': f'drone/{fname}', 'label': 1})

# Фонові звуки
for fname in os.listdir('dataset/no_drone'):
    if fname.endswith('.wav'):
        data.append({'filename': f'no_drone/{fname}', 'label': 0})

# Збереження
df = pd.DataFrame(data)
df.to_csv('dataset/metadata.csv', index=False)

print("metadata.csv успішно створено.")
