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
    config["grid"]["rows"] = rows
    config["grid"]["cols"] = cols
    
    config["map"] = {
        "walls": entity_map["walls"],
        "obstacles": entity_map["obstacles"],
        "exits": entity_map["exits"],
        "fire_sources": entity_map["fire_sources"],
        "agent_start": [list(agent_start)]
    }
    
    if "dynamics" not in config:
        config["dynamics"] = {}
    if "fire_spread_probability" not in config["dynamics"]:
        config["dynamics"]["fire_spread_probability"] = 0.01
    if "smoke_radius" not in config["dynamics"]:
        config["dynamics"]["smoke_radius"] = 1
    
    # Increase max steps for larger maps
    if map_name in ["hospital", "mall"]:
        config["dynamics"]["max_steps"] = 500
    elif map_name == "school":
        config["dynamics"]["max_steps"] = 300
    else:
        config["dynamics"]["max_steps"] = 200
        
    # Reward shaping for Phase 1
    config["rewards"] = {
        "exit_reached": 150.0,
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
