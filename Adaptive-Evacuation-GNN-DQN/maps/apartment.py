from maps.base_map import parse_ascii_map

# Medium-Easy: 14x14 Apartment
APARTMENT_ASCII = [
    "WWWWWWWWWWWWWW",
    "W...W......E.W",
    "W.O.W........W",
    "W.O.W...WWWW.W",
    "W...W...W....W",
    "WW.WW...W..O.W",
    "W.......W..O.W",
    "W.WWWWWWW....W",
    "W.......W....W",
    "W.O.....W.WW.W",
    "W.O.......W..W",
    "W.WWWWWWWWW..W",
    "W....A.......W",
    "WWWWWWWWWWWWWW"
]

def get_apartment_map():
    return parse_ascii_map(APARTMENT_ASCII)
