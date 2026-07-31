import { useState, useEffect } from 'react';

export type Agent = {
  id: number;
  position: [number, number]; // [row, col]
  active: boolean;
};

export type GameState = {
  rows: number;
  cols: number;
  grid: number[][]; // 0=EMPTY, 1=WALL, 2=EXIT, 3=FIRE, 4=SMOKE
  agents: Agent[];
  step: number;
  total_reward: number;
  reason?: string;
};

export function useWebSocket(url: string) {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket;
    let timeoutId: NodeJS.Timeout;

    const connect = () => {
      ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('Connected to WebSocket server');
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data: GameState = JSON.parse(event.data);
          setGameState(data);
        } catch (e) {
          console.error("Failed to parse websocket message", e);
        }
      };

      ws.onclose = () => {
        console.log('Disconnected from WebSocket server. Reconnecting in 2s...');
        setConnected(false);
        timeoutId = setTimeout(connect, 2000);
      };
      
      ws.onerror = (err) => {
        console.error('WebSocket encountered an error:', err);
        ws.close();
      }
    };

    connect();

    return () => {
      clearTimeout(timeoutId);
      if (ws) {
        ws.onclose = null; // Prevent reconnect on unmount
        ws.close();
      }
    };
  }, [url]);

  return { gameState, connected };
}
