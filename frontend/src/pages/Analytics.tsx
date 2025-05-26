import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';

import {
  getFeatureImportance,
  getDistribution,
  fetchHistory,
  getModelComparisons,
  getCheckedMolecules,
} from '../api';

const tabs = [
  'History of Versions',
  'Comparisons of Models',
  'Distribution',
  'Checked Molecules',
  'Feature Importance',
];

const Analytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);

  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [distribution, setDistribution] = useState<Record<string, { mean: number, std: number }> | null>(null);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [checkedMoleculesData, setCheckedMoleculesData] = useState<any[]>([]);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        switch (tabs[activeTab]) {
          case 'History of Versions':
            setHistoryData(await fetchHistory());
            break;

          case 'Comparisons of Models':
            setComparisonData(await getModelComparisons());
            break;

          case 'Distribution':
            setDistribution(await getDistribution());
            break;

          case 'Checked Molecules':
            setCheckedMoleculesData(await getCheckedMolecules());
            break;

          case 'Feature Importance':
            setFeatureImportance(await getFeatureImportance());
            break;
        }
      } catch (err) {
        console.error('Error fetching data for tab:', tabs[activeTab], err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [activeTab]);

  const renderContent = () => {
    const tabName = tabs[activeTab];
    if (loading) return <p>Loading...</p>;

    switch (tabName) {
      case 'History of Versions':
        return (
          <table border={1} cellPadding={5} style={{ width: '100%', textAlign: 'left' }}>
            <thead>
              <tr>
                <th>Version</th>
                <th>Date</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {historyData.map((item, idx) => (
                <tr key={idx}>
                  <td>{item.version}</td>
                  <td>{item.date}</td>
                  <td>{item.accuracy !== null ? (item.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        );

      case 'Comparisons of Models':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="rmse" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'Distribution':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={distribution ? Object.entries(distribution).map(([key, val]) => ({
                name: key,
                mean: val.mean,
                std: val.std,
              })) : []}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="mean" fill="#0088FE" />
              <Bar dataKey="std" fill="#FFBB28" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'Checked Molecules':
        return (
          <table border={1} cellPadding={5} style={{ width: '100%', textAlign: 'left' }}>
            <thead>
              <tr>
                <th>ID</th>
                <th>SMILES</th>
                <th>LogS</th>
              </tr>
            </thead>
            <tbody>
              {checkedMoleculesData.map((item, idx) => (
                <tr key={idx}>
                  <td>{item.id}</td>
                  <td>{item.smiles}</td>
                  <td>{item.logS}</td>
                </tr>
              ))}
            </tbody>
          </table>
        );

      case 'Feature Importance':
        return (
          <div>
            <h3>Feature Importance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureImportance} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="feature" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="importance" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );

      default:
        return <p>Graph for "{tabName}" not implemented.</p>;
    }
  };

  return (
    <div>
      <h2>Analytics</h2>
      <div style={{ display: 'flex', gap: '10px', marginBottom: 20 }}>
        {tabs.map((tab, idx) => (
          <button
            key={tab}
            onClick={() => setActiveTab(idx)}
            style={{ fontWeight: idx === activeTab ? 'bold' : 'normal' }}
          >
            {tab}
          </button>
        ))}
      </div>
      {renderContent()}
    </div>
  );
};

export default Analytics;
