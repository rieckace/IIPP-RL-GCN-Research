import React from 'react';
import type { GameState } from '../hooks/useWebSocket';

interface Props {
  state: GameState;
}

const getCellClass = (val: number) => {
  switch(val) {
    case 1: return 'wall';
    case 3: return 'exit';
    case 5: return 'smoke';
    case 6: return 'fire';
    default: return 'empty';
  }
};

export const GridRenderer: React.FC<Props> = ({ state }) => {
  const { rows, cols, grid, agents } = state;

  return (
    <div 
      className="grid-container"
      style={{ gridTemplateColumns: `repeat(${cols}, 40px)` }}
    >
      {grid.map((row, r) => 
        row.map((cellVal, c) => {
          // Check if any agent is currently on this exact cell
          const activeAgents = agents.filter(a => a.active && a.position[0] === r && a.position[1] === c);
          
          return (
            <div key={`${r}-${c}`} className={`cell ${getCellClass(cellVal)}`}>
              {cellVal === 6 && <span className="fire-emoji">🔥</span>}
              {activeAgents.map(agent => (
                <div key={agent.id} className="agent">
                  <span className="agent-emoji">🏃</span>
                </div>
              ))}
            </div>
          )
        })
      )}
    </div>
  );
};
