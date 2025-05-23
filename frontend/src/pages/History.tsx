import React, { useEffect, useState } from 'react';
import api from '../api';

const History: React.FC = () => {
  const [records, setRecords] = useState<any[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get('/history')
      .then(res => setRecords(res.data))
      .catch(() => setError('Failed to load history'));
  }, []);

  if (error) return <p>{error}</p>;

  return (
    <div>
      <h2>Prediction History</h2>
      {records.length === 0 && <p>No records found</p>}
      <table border={1}>
        <thead>
          <tr>
            <th>SMILES</th>
            <th>LogS</th>
            <th>Mol. Weight</th>
            <th>LogP</th>
            <th>TPSA</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {records.map(r => (
            <tr key={r.id}>
              <td>{r.smiles}</td>
              <td>{r.logS}</td>             
              <td>{r.molecularWeight.toFixed(2)}</td>
              <td>{r.logP.toFixed(2)}</td>
              <td>{r.tpsa.toFixed(2)}</td>
              <td>{new Date(r.timestamp).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default History;
