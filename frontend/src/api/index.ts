import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 60000,  // розумний таймаут, якщо тренування довге
});

export default api;

// --- Основні запити ---
export const predictSolubility = async (smiles: string) => {
  const response = await api.post('/predict', { smiles });
  return response.data;
};

export const fetchHistory = async () => {
  const response = await api.get('/history');
  return response.data;
};

export const trainModel = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/train', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const fetchMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

// --- Аналітика ---

export const getHistoryOfVersions = async () => {
  const res = await axios.get('/api/history-of-versions');
  return res.data;
};

export const getModelComparisons = async () => {
  const res = await axios.get('/api/model-comparisons'); // ✅ правильний шлях
  return res.data;
};

export const getDistribution = async () => {
  const res = await axios.get('/api/distribution'); // ✅ правильний шлях
  return res.data;
};

export const getFeatureImportance = async () => {
  const res = await axios.get('/api/feature-importance'); // ✅ правильний шлях
  return res.data;
};

export const getCheckedMolecules = async () => {
  const res = await axios.get('/api/checked-molecules'); // ✅ правильний шлях
  return res.data;
};
