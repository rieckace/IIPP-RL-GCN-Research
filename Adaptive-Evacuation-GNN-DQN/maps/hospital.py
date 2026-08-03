from maps.base_map import parse_ascii_map

# Hard: 22x22 Hospital
HOSPITAL_ASCII = [
    "WWWWWWWWWWWWWWWWWWWWWW",
    "W....W......W........W",
    "W.O..W.O....W..O...O.W",
    "W....W......W........W",
    "WW..WWWW..WWWWW..WWWWW",
    "W....................W",
    "WWWWWWWW..WWWWWWWW..WW",
    "W....W......W........W",
    "W.O..W.O..F.W..O...O.W",
    "W....W......W........W",
    "WW..WWWW..WWWWW..WWWWW",
    "W.........A..........W",
    "WW..WWWW..WWWWW..WWWWW",
    "W....W......W........W",
    "W....W.O....W..O...O.W",
    "W.O..W......W........W",
    "WW..WWWW..WWWWW..WWWWW",
    "W....................W",
    "WWWWWWWW..WWWWWWWW..WW",
    "W....W......W........W",
    "W.O..W..E...W..O...O.W",
    "WWWWWWWWWWWWWWWWWWWWWW"
]

def get_hospital_map():
    return parse_ascii_map(HOSPITAL_ASCII)
