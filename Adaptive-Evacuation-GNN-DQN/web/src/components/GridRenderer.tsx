import React from 'react';
import type { GameState } from '../hooks/useWebSocket';

interface Props {
  state: GameState;
  onCellClick: (row: number, col: number) => void;
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

export const GridRenderer: React.FC<Props> = ({ state, onCellClick }) => {
  const { rows, cols, grid, agents } = state;

  return (
    <div 
      className="grid-container"
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
    >
      {grid.map((row, r) => 
        row.map((cellVal, c) => {
          const activeAgents = agents.filter(a => a.active && a.position[0] === r && a.position[1] === c);
          const isEmpty = cellVal === 0 || cellVal === 2;
          
          return (
            <div 
              key={`${r}-${c}`} 
              className={`cell ${getCellClass(cellVal)}`}
              onClick={() => {
                if (isEmpty) onCellClick(r, c);
              }}
            >
              {isEmpty && <div className="cell-tooltip">Click to place agent here</div>}
              {cellVal === 6 && <span className="fire-emoji">🔥</span>}
              {activeAgents.map(agent => (
                <div key={agent.id} className="agent-wrapper">
                  <span className="agent-emoji">🏃🏼‍➡️</span>
                </div>
              ))}
            </div>
          )
        })
      )}
    </div>
  );
};
