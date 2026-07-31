from maps.base_map import parse_ascii_map

# Medium: 18x18 School
SCHOOL_ASCII = [
    "WWWWWWWWWWWWWWWWWW",
    "W.W........W.....W",
    "W.W..O..O..W..O..W",
    "W.W........W.....W",
    "W.WWWW..WWWWWW..WW",
    "W................W",
    "WWWWWW..WWWWWWWWWW",
    "W.W........W.....W",
    "W.W..O.....W..O..W",
    "W.W........W.....W",
    "W.WWWW..WWWWWW..WW",
    "W.......A........W",
    "WWWWWW..WWWWWW..WW",
    "W.W........W.....W",
    "W.W........W.....W",
    "W.W..O.....W..O..W",
    "W.......E........W",
    "WWWWWWWWWWWWWWWWWW"
]

def get_school_map():
    return parse_ascii_map(SCHOOL_ASCII)
