import React, { useEffect, useState } from 'react';
import api from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const Metrics: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get('/metrics')
      .then(res => setMetrics(res.data))
      .catch(() => setError('Failed to load metrics'));
  }, []);

  if (error) return <p>{error}</p>;
  if (!metrics) return <p>Loading metrics...</p>;

  // Перетворюємо метрики в масив для графіка
  const chartData = [
    { name: 'R² Score', value: metrics.r2_score },
    { name: 'RMSE', value: metrics.rmse },
    { name: 'MAE', value: metrics.mae }
  ];

  return (
    <div>
      <h2>Model Performance Metrics</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis allowDecimals={true} />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Metrics;
