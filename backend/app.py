from io import StringIO
import os
from datetime import datetime
from shelve import Shelf

import joblib
import json

import pandas as pd
import rdkit
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Crippen import MolMR
from rdkit.Chem.Descriptors import MaxPartialCharge
from rdkit.Chem.EState import EState
from rdkit.Chem.Lipinski import NHOHCount
from rdkit.Chem.MolSurf import TPSA
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcLabuteASA, CalcNumHBD, CalcNumHBA, CalcNumRings, \
    CalcNumAromaticRings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Flask init ===
app = Flask(__name__)
CORS(app)

# === DB Config ===
DB_USER = os.environ.get("DB_USER", "chemuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "chempass")
DB_NAME = os.environ.get("DB_NAME", "chemsolve")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# === DB model ===
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    smiles = db.Column(db.String(200))
    logS = db.Column(db.Float)
    slogp = db.Column(db.Float)
    clogp = db.Column(db.Float)
    xlogp3 = db.Column(db.Float)
    tpsa = db.Column(db.Float)
    asa = db.Column(db.Float)
    psa = db.Column(db.Float)
    hbd = db.Column(db.Integer)
    hba = db.Column(db.Integer)
    nhoh = db.Column(db.Integer)
    bertz_ct = db.Column(db.Float)
    num_rings = db.Column(db.Integer)
    num_aromatic_rings = db.Column(db.Integer)
    fraction_csp3 = db.Column(db.Float)
    mean_estate = db.Column(db.Float)
    sum_partial_logp = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# === Globals ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "latest_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
model = None
model_metrics = {}
feature_names = [
    "SLogP", "CLogP", "XLogP3", "TPSA", "ASA", "PSA",
    "HBD", "HBA", "NHOHCount", "BertzCT", "NumRings", "NumAromaticRings",
    "FractionCsp3", "Mean_EState", "Sum_PartialCharges_LogP"
]
MODEL_DIR = "./models"
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.pkl")
LATEST_METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
def find_previous_files(prefix):
    """Повертає список файлів із префіксом в MODEL_DIR, відсортованих за датою."""
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix)]
    files.sort(reverse=True)
    return files

# === Feature functions ===
def HeavyAtomRatio(mol):
    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    total_atoms = mol.GetNumAtoms()
    return heavy_atoms / total_atoms if total_atoms > 0 else 0.0

def featurize(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Розрахунок індексів та логP-внесків
        estate_indices = EState.EStateIndices(mol)

        return [
        MolLogP(mol),                              # SLogP (замінник логP)
        MolMR(mol),                                # MolMR як поляризовність (аналог CLogP)
        MolMR(mol),                                # дубльоване як XLogP3-заміна
        CalcTPSA(mol),                                         # TPSA — топологічна площа поверхні
        CalcLabuteASA(mol),                                    # ASA — аполярна площа поверхні                            # PSA — VSA по частковим зарядам
        CalcNumHBD(mol),                                       # Кількість донорів водневих зв’язків
        CalcNumHBA(mol),                                       # Кількість акцепторів водневих зв’язків
        NHOHCount(mol),                            # Кількість N–H / O–H груп
        Descriptors.BertzCT(mol),                              # Індекс складності Бертца
        CalcNumRings(mol),                                     # Кількість кілець
        CalcNumAromaticRings(mol),                             # Кількість ароматичних кілець
        TPSA(mol),
        HeavyAtomRatio(mol),
        MaxPartialCharge(mol),
            # Насиченість
        sum(estate_indices) / len(estate_indices) if estate_indices else 0.0  # Середній EState
                                    # Sum_PartialCharges_LogP
        ]
    except Exception as e:
        print(f"[❌ featurize error]: {e}")
        return None




# === Load model if exists ===
def load_model():
    global model, model_metrics
    print("🔍 Checking model path:", model_path)
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("✅ Model loaded.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print("ℹ️ No model file found.")

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                model_metrics = json.load(f)
            print("📊 Metrics loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load metrics: {e}")
    else:
        print("ℹ️ No metrics file found.")

# === Routes ===
@app.route('/')
def index():
    return "✅ ChemSolve backend is running."

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model is not trained yet'}), 503


    data = request.get_json()
    smiles = data.get('smiles')
    print(smiles)
    features = featurize(smiles)
    print(features)

    if features is None:
        return jsonify({'error': 'Invalid SMILES'}), 400

    prediction = round(model.predict([features])[0], 2)

    record = Prediction(
        record=Prediction(
            smiles=smiles,
            logS=prediction,
            slogP=features[0],
            clogP=features[1],
            xlogP3=features[2],
            tpsa=features[3],
            asa=features[4],
            psa=features[5],
            hbd=features[6],
            hba=features[7],
            nhoh_count=features[8],
            bertz_ct=features[9],
            num_rings=features[10],
            num_aromatic_rings=features[11],
            fraction_csp3=features[12],
            mean_estate=features[13],
            sum_partial_charges_logp=features[14]
        )
    )
    db.session.add(record)
    db.session.commit()
    print(f"✅ Prediction saved: {record}")

    print(prediction)
    print(dict(zip(feature_names, features)))

    return jsonify({
        "smiles": smiles,
        "logS": prediction,
        "features": dict(zip(feature_names, features))
    })

@app.route('/history', methods=['GET'])
def history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).limit(50).all()
    return jsonify([
        {
            "id": r.id,
            "smiles": r.smiles,
            "logS": r.logS,
            "slogP": r.slogP,
            "clogP": r.clogP,
            "xlogP3": r.XlogP3,
            "tpsa": r.tspa,
            "asa": r.asa,
            "psa": r.psa,
            "hbd": r.hbd,
            "hba": r.hbd,
            "nhoh_count": r.nhoh_count,
            "bertz_ct": r.bertz_ct,
            "num_rings":r.num_rings,
            "num_aromatic_rings": r.num_aromatic_rings,
            "fraction_csp3": r.fraction_csp3,
            "mean_estate": r.mean_estate,
            "sum_partial_charges_logp": r.sum_partial_charges_logp,
            "timestamp": r.timestamp.isoformat()
        } for r in records
    ])

@app.route('/feature-distribution', methods=['GET'])
def feature_distribution():
    try:
        df = pd.read_sql_table('prediction', con=db.engine)
        dist = {}
        for col in feature_names:
            dist[col] = {
                'mean': round(df[col].mean(), 2),
                'std': round(df[col].std(), 2)
            }
        return jsonify(dist)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    global model_metrics
    if not model_metrics:
        return jsonify({'error': 'Model not trained yet'}), 503
    return jsonify(model_metrics)

@app.route('/train', methods=['POST'])
def train():
    global model, model_metrics
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))

        if 'SMILES' not in df.columns or 'LogS' not in df.columns:
            return jsonify({'error': 'CSV must contain SMILES and LogS columns'}), 400

        df['features'] = df['SMILES'].apply(featurize)
        invalid_count = df['features'].isna().sum()
        if invalid_count > 0:
            print(f"⚠️ Dropped {invalid_count} rows due to invalid SMILES.")
        df = df.dropna(subset=['features'])

        if df.empty:
            return jsonify({'error': 'All rows dropped after featurization. Possibly invalid SMILES.'}), 400

        X = pd.DataFrame(df['features'].to_list(), columns=feature_names)
        y = df['LogS'].values

        if len(X) < 5:
            return jsonify({'error': 'Not enough data to train. At least 5 valid samples required.'}), 400

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        grid = GridSearchCV(RandomForestRegressor(random_state=42), {
            'n_estimators': [100, 150],
            'max_depth': [None, 10, 20]
        }, cv=5)

        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        predictions = model.predict(X_test)
        model_metrics = {
            "r2_score": round(r2_score(y_test, predictions), 4),
            "rmse": round(mean_squared_error(y_test, predictions, squared=False), 4),
            "mae": round(mean_absolute_error(y_test, predictions), 4),
            "best_params": grid.best_params_,
            "feature_importance": dict(zip(feature_names, model.feature_importances_.round(4).tolist()))
        }
        print("📊 Model metrics:", model_metrics)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        with open(METRICS_PATH, "w") as f:
            json.dump(model_metrics, f)

        return jsonify({
            "message": "Model trained and saved successfully",
            "metrics": model_metrics,
            "model_path": model_path
        })

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

class CheckedMolecule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    smiles = db.Column(db.String, unique=True, nullable=False)
    logS = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "smiles": self.smiles,
            "logS": self.logS
        }

@app.route('/api/checked-molecules', methods=['GET'])
def get_checked_molecules():
    smiles_query = request.args.get('smiles')
    if smiles_query:
        # Пошук молекули за точним SMILES
        molecule = CheckedMolecule.query.filter_by(smiles=smiles_query).first()
        if molecule:
            return jsonify([molecule.to_dict()])
        else:
            return jsonify([])  # Якщо не знайдено
    else:
        # Якщо немає параметру - повертаємо всі (або обмежену кількість)
        molecules = CheckedMolecule.query.limit(50).all()
        return jsonify([m.to_dict() for m in molecules])



def find_previous_files(prefix):
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix)]
    files.sort(reverse=True)
    return files

@app.route('/api/history-of-versions', methods=['GET'])
def get_history_of_versions():
    versions = []
    model_files = find_previous_files("previous-")
    for filename in model_files:
        timestamp = filename[len("previous-"):-4]
        metrics_file = f"metrics-{timestamp}.json"
        metrics_path = os.path.join(MODEL_DIR, metrics_file)
        accuracy = None
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                accuracy = metrics.get("r2_score")
        versions.append({
            "version": timestamp,
            "date": timestamp,
            "accuracy": accuracy if accuracy is not None else 0
        })
    return jsonify(versions)

@app.route('/api/model-comparisons', methods=['GET'])
def get_model_comparisons():
    model_files = find_previous_files("previous-")
    comparison = []
    for filename in model_files[:5]:
        timestamp = filename[len("previous-"):-4]
        metrics_file = f"metrics-{timestamp}.json"
        metrics_path = os.path.join(MODEL_DIR, metrics_file)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                comparison.append({
                    "model": timestamp,
                    "rmse": metrics.get("rmse", None)
                })
    return jsonify(comparison)

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    if not os.path.exists(LATEST_METRICS_PATH):
        return jsonify([])
    with open(LATEST_METRICS_PATH, "r") as f:
        metrics = json.load(f)
        importance_dict = metrics.get("feature_importance", {})
        importance = [{"feature": k, "importance": v} for k, v in importance_dict.items()]
    return jsonify(importance)

@app.route('/api/distribution', methods=['GET'])
def get_distribution():
    if not os.path.exists(LATEST_METRICS_PATH):
        return jsonify({})
    with open(LATEST_METRICS_PATH, "r") as f:
        metrics = json.load(f)
        distribution = metrics.get("distribution", None)
        if distribution:
            return jsonify(distribution)
    # Заглушка, якщо в метриках немає розподілу
    return jsonify({
        "MolWeight": {"mean": 300.25, "std": 50.1},
        "LogP": {"mean": 2.5, "std": 0.8},
        "TPSA": {"mean": 75.0, "std": 20.3},
    })

# === Init ===
with app.app_context():
    db.create_all()
    load_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
