import yaml
from environment.evacuation_env import EvacuationEnv
from maps.office import get_office_map
from maps.apartment import get_apartment_map
from maps.school import get_school_map
from maps.hospital import get_hospital_map
from maps.mall import get_mall_map

def make_env(map_name="office", render_mode=None):
    """
    Creates an EvacuationEnv with the specified map.
    """
    # Load default base config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Get the specific map layout
    if map_name == "office":
        rows, cols, entity_map, agent_start = get_office_map()
    elif map_name == "apartment":
        rows, cols, entity_map, agent_start = get_apartment_map()
    elif map_name == "school":
        rows, cols, entity_map, agent_start = get_school_map()
    elif map_name == "hospital":
        rows, cols, entity_map, agent_start = get_hospital_map()
    elif map_name == "mall":
        rows, cols, entity_map, agent_start = get_mall_map()
    else:
        raise ValueError(f"Unknown map name: {map_name}")
        
    # Inject map into config
    # Pad everything to a fixed 30x30 grid for universal generalization
    MAX_SIZE = 30
    row_offset = (MAX_SIZE - rows) // 2
    col_offset = (MAX_SIZE - cols) // 2
    
    # Generate background of walls for the 30x30 grid
    padded_walls = []
    for r in range(MAX_SIZE):
        for c in range(MAX_SIZE):
            if r < row_offset or r >= row_offset + rows or c < col_offset or c >= col_offset + cols:
                padded_walls.append([r, c])
                
    # Offset the inner map entities
    for w in entity_map["walls"]: padded_walls.append([w[0] + row_offset, w[1] + col_offset])
    shifted_obstacles = [[o[0] + row_offset, o[1] + col_offset] for o in entity_map["obstacles"]]
    shifted_exits = [[e[0] + row_offset, e[1] + col_offset] for e in entity_map["exits"]]
    shifted_fires = [[f[0] + row_offset, f[1] + col_offset] for f in entity_map["fire_sources"]]
    shifted_agent = [agent_start[0] + row_offset, agent_start[1] + col_offset]
    
    config["grid"]["rows"] = MAX_SIZE
    config["grid"]["cols"] = MAX_SIZE
    
    config["map"] = {
        "walls": padded_walls,
        "obstacles": shifted_obstacles,
        "exits": shifted_exits,
        "fire_sources": shifted_fires,
        "agent_start": [shifted_agent]
    }
    
    # Phase 1: Disable dynamic fire spread
    if "dynamics" not in config:
        config["dynamics"] = {}
    config["dynamics"]["fire_spread_probability"] = 0.0
    config["dynamics"]["smoke_radius"] = 0
    
    # Increase max steps for larger maps
    if map_name in ["hospital", "mall"]:
        config["dynamics"]["max_steps"] = 500
    elif map_name == "school":
        config["dynamics"]["max_steps"] = 300
    else:
        config["dynamics"]["max_steps"] = 200
        
    # Reward shaping for Phase 1
    config["rewards"] = {
        "exit_reached": 100.0,
        "fire_hit": -50.0,
        "smoke_step": -10.0, # Not used in Phase 1
        "wall_bump": -10.0,  # As requested: -10 for wall/obstacle
        "normal_step": -0.1, # As requested: -0.1 for valid movement
        "stay_penalty": -2.0,# As requested: -2 for staying
        "exit_progress_scale": 1.0 # DENSE REWARD: +1 for moving closer, -1 for moving away
    }
    
    env = EvacuationEnv(config, render_mode=render_mode)
    env._config = config
    return env
