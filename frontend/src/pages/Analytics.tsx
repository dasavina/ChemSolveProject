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

  // useEffect(() => {
  //   const loadData = async () => {
  //     setLoading(true);
  //     try {
  //       switch (tabs[activeTab]) {
  //         case 'History of Versions':
  //           setHistoryData(await fetchHistory());
  //           break;
  //         case 'Comparisons of Models':
  //           setComparisonData(await getModelComparisons());
  //           break;
  //         case 'Distribution':
  //           setDistribution(await getDistribution());
  //           break;
  //         case 'Checked Molecules':
  //           setCheckedMoleculesData(await getCheckedMolecules());
  //           break;
  //         case 'Feature Importance':
  //           setFeatureImportance(await getFeatureImportance());
  //           break;
  //       }
  //     } catch (err) {
  //       console.error(err);
  //     } finally {
  //       setLoading(false);
  //     }
  //   };
  //   loadData();
  // }, [activeTab]);

useEffect(() => {
  const loadData = async () => {
    setLoading(true);
    try {
      switch (tabs[activeTab]) {
        case 'History of Versions':
          setHistoryData([
            { version: '20250520-141230', date: '2025-05-20', accuracy: 0.912 },
            { version: '20250518-095820', date: '2025-05-18', accuracy: 0.887 },
          ]);
          break;

        case 'Comparisons of Models':
          setComparisonData([
            { model: '20250520-141230', rmse: 0.152 },
            { model: '20250518-095820', rmse: 0.179 },
          ]);
          break;

        case 'Distribution':
          setDistribution({
            MolWeight: { mean: 312.6, std: 48.2 },
            LogP: { mean: 2.85, std: 0.92 },
            TPSA: { mean: 72.4, std: 18.7 },
            HBD: { mean: 1.5, std: 0.7 },
            HBA: { mean: 4.6, std: 1.9 },
          });
          break;

        case 'Checked Molecules':
          setCheckedMoleculesData([
        {
            "id": 1,
            "smiles": "c1ccccc1",
            "logS": -1.53,
            "slogP": 78.11,
            "clogP": 1.69,
            "xlogP3": 0.00,
            "tpsa": 12.89,
            "asa": 35.11,
            "psa": 12.89,
            "hbd": 0,
            "hba": 0,
            "nhoh_count": 0,
            "bertz_ct": 215.3,
            "num_rings": 1,
            "num_aromatic_rings": 1,
            "fraction_csp3": 0.0,
            "mean_estate": 0.25,
            "sum_partial_charges_logp": -0.08,
            "timestamp": "2025-05-23T05:14:36"
        },
        {
            "id": 2,
            "smiles": "CCCC",
            "logS": -2.43,
            "slogP": 58.12,
            "clogP": 1.81,
            "xlogP3": 0.00,
            "tpsa": 0.0,
            "asa": 32.55,
            "psa": 0.0,
            "hbd": 0,
            "hba": 0,
            "nhoh_count": 0,
            "bertz_ct": 185.7,
            "num_rings": 0,
            "num_aromatic_rings": 0,
            "fraction_csp3": 1.0,
            "mean_estate": -0.15,
            "sum_partial_charges_logp": -0.12,
            "timestamp": "2025-05-23T04:47:00"
        },
        {
            "id": 3,
            "smiles": "CCC",
            "logS": -1.87,
            "slogP": 44.10,
            "clogP": 1.42,
            "xlogP3": 0.00,
            "tpsa": 0.0,
            "asa": 25.88,
            "psa": 0.0,
            "hbd": 0,
            "hba": 0,
            "nhoh_count": 0,
            "bertz_ct": 162.4,
            "num_rings": 0,
            "num_aromatic_rings": 0,
            "fraction_csp3": 1.0,
            "mean_estate": -0.18,
            "sum_partial_charges_logp": -0.10,
            "timestamp": "2025-05-22T20:33:50"
        },
        {
            "id": 4,
            "smiles": "C(CO)O",
            "logS": 0.78,
            "slogP": 62.07,
            "clogP": -1.03,
            "xlogP3": 40.46,
            "tpsa": 57.53,
            "asa": 48.12,
            "psa": 57.53,
            "hbd": 2,
            "hba": 2,
            "nhoh_count": 2,
            "bertz_ct": 145.6,
            "num_rings": 0,
            "num_aromatic_rings": 0,
            "fraction_csp3": 1.0,
            "mean_estate": 0.10,
            "sum_partial_charges_logp": 0.03,
            "timestamp": "2025-05-22T17:24:06"
        }
    ]);
          break;

        case 'Feature Importance':
          setFeatureImportance([
  { "feature": "logS", "importance": 0.25 },
  { "feature": "slogP", "importance": 0.18 },
  { "feature": "clogP", "importance": 0.15 },
  { "feature": "xlogP3", "importance": 0.12 },
  { "feature": "tpsa", "importance": 0.08 },
  { "feature": "asa", "importance": 0.05 },
  { "feature": "psa", "importance": 0.03 },
  { "feature": "hbd", "importance": 0.01 },
  { "feature": "hba", "importance": 0.01 },
  { "feature": "nhoh_count", "importance": 0.02 },
  { "feature": "bertz_ct", "importance": 0.02 },
  { "feature": "num_rings", "importance": 0.01 },
  { "feature": "num_aromatic_rings", "importance": 0.02 },
  { "feature": "fraction_csp3", "importance": 0.01 },
  { "feature": "mean_estate", "importance": 0.03 },
  { "feature": "sum_partial_charges_logp", "importance": 0.02 }


          ]);
          break;
      }
    } catch (err) {
      console.error(err);
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
