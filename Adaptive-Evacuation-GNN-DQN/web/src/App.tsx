import React, { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { GridRenderer } from './components/GridRenderer';

export function App() {
  const { gameState, connected } = useWebSocket('ws://localhost:8000/stream');
  const [modelType, setModelType] = useState('dqn');
  const [gridSize, setGridSize] = useState(10);
  
  const handleApply = async () => {
    try {
      await fetch('http://localhost:8000/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelType, grid_size: gridSize })
      });
    } catch (e) {
      console.error("Failed to reconfigure server", e);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>MARL Evacuation Dashboard</h1>
        <p>
          Status: {connected ? <span style={{color: '#10B981'}}>Connected (Live)</span> : <span style={{color: '#EF4444'}}>Disconnected</span>}
        </p>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <h3>Control Panel</h3>
          
          <div className="control-group">
            <label>AI Model</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="dqn">Double DQN (Baseline)</option>
              <option value="gnn">Standard GNN</option>
              <option value="hybrid">Hybrid GNN-A*</option>
              <option value="marl">MARL (3 Agents)</option>
            </select>
          </div>

          <div className="control-group">
            <label>Grid Size</label>
            <select 
              value={gridSize} 
              onChange={(e) => setGridSize(parseInt(e.target.value))}
              disabled={modelType === 'dqn'} // DQN hardcoded to 10x10 architecture
            >
              <option value={10}>10x10 (Trained)</option>
              <option value={15}>15x15 (Zero-Shot)</option>
            </select>
            {modelType === 'dqn' && <small>DQN is locked to 10x10</small>}
          </div>

          <button className="btn-primary" onClick={handleApply}>
            Apply & Restart
          </button>
          
          {gameState && (
            <div className="stats-panel-mini">
              <div className="stat-item">
                <span className="stat-label">Step</span>
                <span className="stat-value">{gameState.step}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Total Reward</span>
                <span className="stat-value" style={{ color: gameState.total_reward < 0 ? '#EF4444' : '#10B981'}}>
                  {gameState.total_reward.toFixed(1)}
                </span>
              </div>
            </div>
          )}
        </aside>

        <main className="dashboard">
          {gameState ? (
            <GridRenderer state={gameState} />
          ) : (
            <div>Waiting for simulation data...</div>
          )}
        </main>

        <aside className="right-sidebar">
          <div className="legend-panel">
            <h4>Legend</h4>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#1A212D'}}></div>
              <span>Corridor / Empty</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#334155'}}></div>
              <span>Wall</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#EF4444'}}>🔥</div>
              <span>Fire</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#10B981'}}></div>
              <span>Exit</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#3B82F6', borderRadius: '50%'}}></div>
              <span>Agent</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
