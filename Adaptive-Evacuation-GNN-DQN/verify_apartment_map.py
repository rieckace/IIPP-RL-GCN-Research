import os
import sys

sys.path.insert(0, os.getcwd())
from maps.apartment import get_apartment_map

rows, cols, entity_map, agent_start = get_apartment_map()
print('rows', rows)
print('cols', cols)
print('fire_sources', entity_map['fire_sources'])
print('agent_start', agent_start)
