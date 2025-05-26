import React, { useState } from 'react';
import api from '../api';

const featureNames = [
  "SLogP", "CLogP", "XLogP3", "TPSA", "ASA", "PSA",
  "HBD", "HBA", "NHOHCount", "BertzCT", "NumRings",
  "NumAromaticRings", "FractionCsp3", "Mean_EState", "Sum_PartialCharges_LogP"
];

const Predict: React.FC = () => {
  const [smiles, setSmiles] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const onSubmit = async () => {
    setError('');
    setResult(null);
    try {
      const response = await api.post('/predict', { smiles });
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || 'Prediction failed');
    }
  };

  return (
    <div>
      <h2>Predict Solubility</h2>
      <input
        type="text"
        placeholder="Enter SMILES string"
        value={smiles}
        onChange={e => setSmiles(e.target.value)}
      />
      <button onClick={onSubmit}>Predict</button>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {result && (
        <div>
          <h3>Prediction Results</h3>
          <p><strong>SMILES:</strong> {result.smiles}</p>
          <p><strong>LogS:</strong> {result.logS}</p>
          <h4>Features:</h4>
          <ul>
            {result.features.map((val: {name: string; value: number}, idx: number) => (
    <li key={idx}>
      <>
        <strong>{val.name || featureNames[idx] || `Feature ${idx + 1}`}:</strong> {val.value}
      </>
    </li>
  ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Predict;
