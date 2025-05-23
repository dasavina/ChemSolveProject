import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => (
  <div>
    <h1>Welcome to ChemSolve</h1>
    <p>Predict aqueous solubility of chemical compounds using SMILES strings.</p>
    <nav>
      <ul>
        <li><Link to="/predict">Predict</Link></li>
        <li><Link to="/train">Train Model</Link></li>
        <li><Link to="/metrics">Metrics</Link></li>
        <li><Link to="/history">History</Link></li>
        <li><Link to="/analytics">Analytics</Link></li>
      </ul>
    </nav>
  </div>
);

export default Home;
