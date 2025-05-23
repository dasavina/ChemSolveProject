import os
import pandas as pd
import numpy as np
import joblib
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, EState
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Шлях до CSV-файлу ===
data_path = 'D:/курсова/LogP_LogS.csv'

# === Завантаження датасету ===
df = pd.read_csv(data_path)

# Перевірка наявності необхідних колонок
if 'SMILES' not in df.columns or 'LogS' not in df.columns:
    raise ValueError("CSV повинен містити колонки 'SMILES' та 'LogS'")

# === Функція для витягнення дескрипторів ===
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        estate_indices = np.array(EState.EStateIndices(mol))
        return [
            Crippen.MolLogP(mol),  # SLogP
            Descriptors.MolLogP(mol),  # CLogP
            Crippen.MolLogP(mol),  # XLogP3 (той самий для спрощення)
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcLabuteASA(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.BertzCT(mol),
            rdMolDescriptors.CalcNumRings(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            np.mean(estate_indices) if len(estate_indices) > 0 else 0.0,
            np.sum(estate_indices) if len(estate_indices) > 0 else 0.0
        ]
    except Exception as e:
        print(f"❌ Failed to extract features for {smiles}: {e}")
        return None


# === Обчислення ознак ===
df['features'] = df['SMILES'].apply(extract_features)
df = df.dropna(subset=['features'])

# Побудова матриці ознак і цільової змінної
feature_names = [
    "SLogP", "CLogP", "XLogP3", "TPSA", "ASA", "PSA",
    "HBD", "HBA", "NHOHCount", "BertzCT", "NumRings", "NumAromaticRings",
    "FractionCsp3", "Mean_EState", "Sum_EState"
]
X = pd.DataFrame(df['features'].tolist(), columns=feature_names)
y = df['LogS'].values

# === Розділення даних ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Навчання моделі ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Прогнозування ===
y_pred = model.predict(X_test)

# === Обчислення метрик ===
metrics = {
    "r2_score": round(r2_score(y_test, y_pred), 4),
    "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    "mae": round(mean_absolute_error(y_test, y_pred), 4),
    "feature_importance": dict(zip(feature_names, model.feature_importances_.round(4).tolist()))
}

# === Створення директорії ===
os.makedirs('models', exist_ok=True)

# === Збереження моделі ===
joblib.dump(model, 'models/latest_model.pkl')

# === Збереження метрик ===
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Модель успішно навчено та збережено")
print("📊 Метрики:")
print(json.dumps(metrics, indent=2))
