import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Train from './pages/Train';
import Metrics from './pages/Metrics';
import History from './pages/History';
import Analytics from './pages/Analytics';

const App: React.FC = () => (
  <Router>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<Predict />} />
      <Route path="/train" element={<Train />} />
      <Route path="/metrics" element={<Metrics />} />
      <Route path="/history" element={<History />} />
      <Route path="/analytics" element={<Analytics />} />
    </Routes>
  </Router>
);

export default App;
