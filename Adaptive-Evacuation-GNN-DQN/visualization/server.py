import os
import sys
import asyncio
import json
import logging
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.evacuation_env import EvacuationEnv
from environment.marl_env import MARLEvacuationEnv
from environment.wrappers import HybridObservationWrapper, MARLGraphObservationWrapper
from utils.config_loader import load_config

from models.dqn.trainer import DQNAgent
from models.gnn.trainer import GNNDQNAgent
from models.hybrid.trainer import HybridGNNDQNAgent
from models.marl.trainer import MARLAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("visualization_server")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Simulation State
class SimState:
    def __init__(self):
        self.model_type = "dqn"
        self.grid_size = 10
        self.reset_requested = True

sim_state = SimState()

class ConfigRequest(BaseModel):
    model: str
    grid_size: int

@app.post("/configure")
async def configure_simulation(req: ConfigRequest):
    sim_state.model_type = req.model
    sim_state.grid_size = req.grid_size
    sim_state.reset_requested = True
    return {"status": "ok"}

def _coord_to_id(r, c, cols):
    if r == -1: return -1
    return r * cols + c

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected.")
    
    env = None
    base_env = None
    agent = None
    is_marl = False
    
    try:
        while True:
            # Re-initialize if requested
            if sim_state.reset_requested:
                logger.info(f"Re-initializing simulation: {sim_state.model_type} on {sim_state.grid_size}x{sim_state.grid_size}")
                sim_state.reset_requested = False
                
                # Load config
                config = load_config("configs/default.yaml", validate=False)
                if sim_state.model_type == "marl":
                    config = load_config("configs/marl.yaml", validate=False)
                    
                # Override grid size
                config["grid"]["rows"] = sim_state.grid_size
                config["grid"]["cols"] = sim_state.grid_size
                
                is_marl = (sim_state.model_type == "marl")
                
                if is_marl:
                    base_env = MARLEvacuationEnv(config)
                    env = MARLGraphObservationWrapper(base_env)
                    agent = MARLAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "marl", "best_model.pt")
                elif sim_state.model_type == "hybrid":
                    base_env = EvacuationEnv(config)
                    env = HybridObservationWrapper(base_env)
                    agent = HybridGNNDQNAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "hybrid", "best_model.pt")
                elif sim_state.model_type == "gnn":
                    base_env = EvacuationEnv(config)
                    env = HybridObservationWrapper(base_env) # Assuming GNN uses graph obs
                    agent = GNNDQNAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "gnn", "best_model.pt")
                else: # dqn
                    base_env = EvacuationEnv(config)
                    env = base_env # DQN takes flat obs directly
                    agent = DQNAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "dqn", "best_model.pt")
                
                # Load checkpoint if exists
                if os.path.exists(ckpt) and hasattr(agent, "load_checkpoint"):
                    try:
                        agent.load_checkpoint(ckpt)
                        agent.epsilon = 0.05 # Tiny epsilon to prevent deadlocks
                        logger.info(f"Loaded {ckpt}")
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint {ckpt}: {e}. Falling back to untrained agent.")
                        agent.epsilon = 1.0
                else:
                    logger.warning(f"No checkpoint found at {ckpt}. Using random agent.")
                    agent.epsilon = 1.0
                    
            # Run Episode Loop
            obs, info = env.reset()
            active_agents = info.get("active_agents", [True]) if is_marl else [info.get("reason", "") != "reached_exit" and info.get("reason", "") != "hit_fire"]
            
            await send_state(websocket, base_env, info, active_agents, is_marl)
            await asyncio.sleep(1.0)
            
            terminated, truncated = False, False
            while not (terminated or truncated):
                if sim_state.reset_requested:
                    break # Break inner loop to trigger re-init
                
                # Act
                if is_marl:
                    agent_node_ids = [_coord_to_id(r, c, base_env.cols) for (r, c) in info["agent_positions"]]
                    action = agent.act(obs, active_agents, agent_node_ids, explore=(agent.epsilon > 0))
                elif sim_state.model_type == "dqn":
                    action = agent.act(obs, explore=(agent.epsilon > 0))
                else:
                    action = agent.act(obs, explore=(agent.epsilon > 0))
                    
                obs, rewards, terminated, truncated, info = env.step(action)
                
                if is_marl:
                    active_agents = info["active_agents"]
                else:
                    actv = not (terminated or truncated)
                    if terminated and info.get("reason") in ["reached_exit", "hit_fire"]:
                        actv = False
                    else:
                        actv = True
                    active_agents = [actv]
                    
                await send_state(websocket, base_env, info, active_agents, is_marl)
                await asyncio.sleep(0.3) # Faster simulation speed
                
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in websocket: {e}")

async def send_state(websocket: WebSocket, base_env, info: dict, active_agents: list, is_marl: bool):
    grid_data = []
    for r in range(base_env.rows):
        row_data = []
        for c in range(base_env.cols):
            cell = base_env.grid.get_cell(r, c)
            row_data.append(cell.value)
        grid_data.append(row_data)
        
    if is_marl:
        agents = [{"id": i, "position": pos, "active": act} for i, (pos, act) in enumerate(zip(info["agent_positions"], active_agents))]
    else:
        pos = info.get("agent_position", (-1, -1))
        agents = [{"id": 0, "position": pos, "active": active_agents[0]}]
    
    payload = {
        "rows": base_env.rows,
        "cols": base_env.cols,
        "grid": grid_data,
        "agents": agents,
        "step": info.get("step", 0),
        "total_reward": info.get("total_reward", 0.0)
    }
    
    await websocket.send_text(json.dumps(payload))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("visualization.server:app", host="0.0.0.0", port=8000, reload=True)
