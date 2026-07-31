from maps.base_map import parse_ascii_map

# Easy: 10x10 Office
OFFICE_ASCII = [
    "WWWWWWWWWW",
    "W........W",
    "W..O.....W",
    "W..O.....W",
    "W........W",
    "W.....O..W",
    "W.....O..W",
    "W....A...W",
    "W....E...W",
    "WWWWWWWWWW"
]

def get_office_map():
    return parse_ascii_map(OFFICE_ASCII)
