import os
import sys
import asyncio
import json
import logging
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# Global Simulation State
class SimState:
    def __init__(self):
        self.model_type = "dqn"
        self.map_name = "office"
        self.agent_start = None
        self.reset_requested = True

sim_state = SimState()

class ConfigRequest(BaseModel):
    model: str
    map_name: str
    agent_start: list[int] | None = None

@app.post("/configure")
async def configure_simulation(req: ConfigRequest):
    sim_state.model_type = req.model
    sim_state.map_name = req.map_name
    sim_state.agent_start = req.agent_start
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
        from environment.make_env import make_env
        while True:
            # Re-initialize if env is None or reset is requested
            if env is None or sim_state.reset_requested:
                logger.info(f"Re-initializing simulation: {sim_state.model_type} on {sim_state.map_name}")
                sim_state.reset_requested = False
                
                # Use the new benchmark map factory
                base_env = make_env(sim_state.map_name)
                
                if sim_state.agent_start is not None:
                    logger.info(f"Overriding agent start position to: {sim_state.agent_start}")
                    base_env.unwrapped._agent_starts = [list(sim_state.agent_start)]
                
                # Load correct config for the specific model type to match agent network layout
                if sim_state.model_type == "gnn":
                    config = load_config("configs/gnn.yaml", validate=False)
                elif sim_state.model_type == "hybrid":
                    config = load_config("configs/hybrid.yaml", validate=False)
                elif sim_state.model_type == "marl":
                    config = load_config("configs/marl.yaml", validate=False)
                else:
                    config = load_config("configs/default.yaml", validate=False)
                
                is_marl = (sim_state.model_type == "marl")
                
                if sim_state.model_type == "dqn":
                    env = base_env # DQN takes flat obs directly
                    agent = DQNAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "dqn", "best_model.pt")
                elif sim_state.model_type == "gnn":
                    from environment.wrappers import GraphObservationWrapper
                    env = GraphObservationWrapper(base_env)
                    agent = GNNDQNAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "gnn", "best_model.pt")
                elif sim_state.model_type == "hybrid":
                    from environment.wrappers import HybridObservationWrapper
                    env = HybridObservationWrapper(base_env)
                    agent = HybridGNNDQNAgent(config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "hybrid", "best_model.pt")
                elif sim_state.model_type == "marl":
                    from environment.wrappers import MARLGraphObservationWrapper
                    env = MARLGraphObservationWrapper(base_env)
                    marl_config = load_config("configs/marl.yaml", validate=False)
                    agent = MARLAgent(marl_config)
                    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "marl", "best_model.pt")
                
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

            # Hold the terminal state until an explicit reset is requested.
            # This prevents the simulation from immediately restarting after
            # a success/failure frame, which makes the final outcome visible.
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

            # Keep the final state on screen until the client explicitly
            # requests a reset from the configure endpoint.
            if terminated or truncated:
                if is_marl:
                    logger.info("Episode ended; waiting for explicit reset request.")
                else:
                    logger.info(f"Episode ended with reason={info.get('reason')}; waiting for explicit reset request.")
                while not sim_state.reset_requested:
                    await asyncio.sleep(0.2)
                
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
            row_data.append(int(cell))
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
        "total_reward": info.get("total_reward", 0.0),
        "reason": info.get("reason", ""),
    }
    
    await websocket.send_text(json.dumps(payload))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("visualization.server:app", host="0.0.0.0", port=8000, reload=True)
