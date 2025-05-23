import React, { useState } from 'react';
import api from '../api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const Train: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState('');
  const [metrics, setMetrics] = useState<{
    r2_score?: number[];
    rmse?: number[];
    mae?: number[];
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
      setMessage('');
      setMetrics(null);
    }
  };

  const onSubmit = async () => {
    if (!file) {
      setMessage('Please select a CSV file');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setMessage('');
    setMetrics(null);

    try {
      const response = await api.post('/train', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 1000000,
      });

      setMessage(response.data.message || 'Training completed');
      if (response.data.metrics) {
        setMetrics(response.data.metrics);
      }
    } catch (error: any) {
      console.log(error);
      setMessage(error.response?.data?.error || 'Training failed');
    } finally {
      setLoading(false);
    }
  };

  // Формуємо дані для графіка
  const chartData = (() => {
    if (!metrics || !metrics.r2_score || !metrics.rmse || !metrics.mae) return [];

    const minLength = Math.min(
      metrics.r2_score.length,
      metrics.rmse.length,
      metrics.mae.length
    );

    return Array.from({ length: minLength }, (_, idx) => ({
      epoch: idx + 1,
      R2: metrics.r2_score?.[idx],
      RMSE: metrics.rmse?.[idx],
      MAE: metrics.mae?.[idx],
    }));
  })();

  return (
    <div>
      <h2>Train Model</h2>
      <input type="file" accept=".csv" onChange={onFileChange} />
      <button onClick={onSubmit} disabled={loading}>
        {loading ? 'Training...' : 'Start Training'}
      </button>
      {message && <p>{message}</p>}

      {chartData.length > 0 && (
        <div style={{ width: '100%', height: 300, marginTop: 20 }}>
          <ResponsiveContainer>
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="R2" stroke="#8884d8" />
              <Line type="monotone" dataKey="RMSE" stroke="#82ca9d" />
              <Line type="monotone" dataKey="MAE" stroke="#ff7300" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default Train;
