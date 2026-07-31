import numpy as np

def parse_ascii_map(ascii_lines):
    """
    Parses a list of strings representing the grid layout into a format compatible
    with the Grid environment.

    Legend:
    'W' : Wall
    'O' : Obstacle
    'E' : Exit
    'A' : Agent Start
    '.' : Empty
    
    Returns:
        rows, cols, entity_map, agent_start
    """
    rows = len(ascii_lines)
    cols = len(ascii_lines[0]) if rows > 0 else 0
    
    entity_map = {
        "walls": [],
        "exits": [],
        "obstacles": [],
        "fire_sources": [],
        "sensors": []
    }
    
    agent_start = (0, 0)
    
    for r, line in enumerate(ascii_lines):
        # Ignore leading/trailing whitespace but keep string length correct
        line = line.strip() 
        for c, char in enumerate(line):
            if char == 'W':
                entity_map["walls"].append([r, c])
            elif char == 'O':
                entity_map["obstacles"].append([r, c])
            elif char == 'E':
                entity_map["exits"].append([r, c])
            elif char == 'A':
                agent_start = (r, c)
            # '.' is implicitly empty, which is the default in Grid
                
    return rows, cols, entity_map, agent_start
