import { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { GridRenderer } from './components/GridRenderer';

export function App() {
  const { gameState, connected } = useWebSocket('ws://localhost:8000/stream');
  const [modelType, setModelType] = useState('gnn');
  const [mapName, setMapName] = useState('office');
  
  const handleApply = async () => {
    try {
      await fetch('http://localhost:8000/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelType, map_name: mapName })
      });
    } catch (e) {
      console.error("Failed to apply config", e);
    }
  };

  const handleReset = async () => {
    try {
      await fetch('http://localhost:8000/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelType, map_name: mapName })
      });
    } catch (e) {
      console.error("Failed to reset simulation", e);
    }
  };

  // Place agent dynamically
  const handlePlaceAgent = async (row: number, col: number) => {
    if (!connected) return;
    try {
      await fetch(`http://localhost:8000/spawn_agent?row=${row}&col=${col}`, {
        method: 'POST'
      });
    } catch (e) {
      console.error("Failed to place agent", e);
    }
  };

  const outcome = gameState?.reason;
  const outcomeLabel = outcome === 'reached_exit'
    ? 'Success - agent reached the exit'
    : outcome === 'hit_fire'
      ? 'Failure - agent was caught by fire'
      : outcome === 'max_steps_exceeded'
        ? 'Timeout - max steps exceeded'
        : outcome === 'reset'
          ? 'Ready'
          : 'Evacuating...';

  const fireCount = gameState?.grid.flat().filter(c => c === 6).length || 0;
  const smokeCount = gameState?.grid.flat().filter(c => c === 5).length || 0;

  return (
    <div className="app-container">
      <header>
        <h1>🏢 Adaptive Evacuation GNN-DQN</h1>
        <div className={`status-pill ${connected ? 'live' : 'offline'}`}>
          {connected ? 'Connected (Live)' : 'Disconnected'}
        </div>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <h3>Control Center</h3>
          
          <div className="control-group">
            <label>AI Model</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="gnn">GNN-DQN Model</option>
              <option value="dqn">Double DQN (Baseline)</option>
              <option value="hybrid">Hybrid GNN-A*</option>
              <option value="marl">MARL (3 Agents)</option>
            </select>
          </div>

          <div className="control-group">
            <label>Map Layout</label>
            <select 
              value={mapName} 
              onChange={(e) => setMapName(e.target.value)}
              disabled={!connected}
            >
              <option value="office">Office (Easy - 10x10)</option>
              <option value="apartment">Apartment (Med - 14x14)</option>
              <option value="school">School (Hard - 18x18)</option>
              <option value="hospital">Hospital (Harder - 22x22)</option>
              <option value="mall">Mall (Extreme - 30x30)</option>
            </select>
          </div>

          <button className="btn-primary" onClick={handleApply}>
            Apply & Restart
          </button>

          <button
            className="btn-secondary"
            onClick={handleReset}
            disabled={!connected}
          >
            Reset Environment
          </button>

          {gameState && (
            <div className={`status-box ${outcome === 'reached_exit' ? 'success' : outcome === 'hit_fire' || outcome === 'max_steps_exceeded' ? 'failure' : 'neutral'}`}>
              {outcomeLabel}
            </div>
          )}

          <div className="legend-panel">
            <label style={{fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase'}}>Grid Legend</label>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#1e293b'}}></div>
              <span>Corridor / Path</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#475569'}}></div>
              <span>Wall</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#10b981'}}></div>
              <span>Exit Zone</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#334155'}}>〰</div>
              <span>Smoke Layer</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#ef4444'}}>🔥</div>
              <span>Active Fire</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{backgroundColor: '#3b82f6', borderRadius: '50%'}}>🏃🏼‍➡️</div>
              <span>Evacuation Agent</span>
            </div>
          </div>
        </aside>

        <main className="dashboard" style={{position: 'relative'}}>
          {gameState ? (
            <>
              <div style={{
                width: '100%', height: '100%', 
                opacity: (outcome === 'reached_exit' || outcome === 'hit_fire' || outcome === 'max_steps_exceeded') ? 0.3 : 1,
                transition: 'opacity 0.5s ease',
                display: 'flex', justifyContent: 'center', alignItems: 'center'
              }}>
                <GridRenderer state={gameState} onCellClick={handlePlaceAgent} />
              </div>
              
              {(outcome === 'reached_exit' || outcome === 'hit_fire' || outcome === 'max_steps_exceeded') && (
                <div className="outcome-overlay" style={{
                  position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                  display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
                  zIndex: 50, borderRadius: '12px',
                  animation: 'fadeIn 0.5s ease forwards'
                }}>
                  <h2 style={{
                    fontSize: '4rem', margin: 0, letterSpacing: '2px',
                    color: outcome === 'reached_exit' ? '#10b981' : '#ef4444',
                    textShadow: `0 0 30px ${outcome === 'reached_exit' ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)'}`
                  }}>
                    {outcome === 'reached_exit' ? 'SUCCESS' : 'FAILED'}
                  </h2>
                  <p style={{fontSize: '1.2rem', color: '#cbd5e1', margin: '20px 0 40px'}}>
                    {outcome === 'reached_exit' ? 'The agent successfully evacuated the building!' : 'The agent failed to evacuate the building.'}
                  </p>
                  <button className="btn-primary" onClick={handleReset} style={{fontSize: '1.1rem', padding: '12px 32px', boxShadow: '0 4px 20px rgba(59,130,246,0.4)'}}>
                    Run Another Episode
                  </button>
                </div>
              )}
            </>
          ) : (
            <div>Waiting for simulation data...</div>
          )}
        </main>

        <aside className="right-sidebar">
          <h3>Simulation Statistics</h3>
          
          {gameState ? (
            <>
              <div className="stat-card">
                <div className="stat-card-title">Step Index</div>
                <div className="stat-card-value">{gameState.step}</div>
              </div>

              <div className="stat-card">
                <div className="stat-card-title">Cumulative Reward</div>
                <div className={`stat-card-value ${gameState.total_reward < 0 ? 'negative' : 'positive'}`}>
                  {gameState.total_reward > 0 ? '+' : ''}{gameState.total_reward.toFixed(1)}
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-card-title">Active Hazards</div>
                <div style={{fontSize: '0.9rem', marginTop: '8px', color: '#f8fafc'}}>
                  🔥 Fire Cells: <span style={{color: '#ef4444', fontWeight: 'bold'}}>{fireCount}</span>
                  <br/>
                  〰 Smoke: <span style={{color: '#94a3b8', fontWeight: 'bold'}}>{smokeCount}</span>
                </div>
              </div>
            </>
          ) : (
            <div style={{color: '#94a3b8', fontSize: '0.9rem'}}>No data available</div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default App;
