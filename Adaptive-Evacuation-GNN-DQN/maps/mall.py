from maps.base_map import parse_ascii_map

# Very Hard: 26x26 Mall
MALL_ASCII = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W.....W............W.....W",
    "W..O..W...O....O...W..O..W",
    "W.....W............W.....W",
    "W.....WWWW......WWWW.....W",
    "W........................W",
    "WWWW.......O..O.......WWWW",
    "W.W....................W.W",
    "W.W....WWWW....WWWW....W.W",
    "W.W....W..W....W..W....W.W",
    "W.W....W.OW....WO.W....W.W",
    "W.W....WWWW....WWWW....W.W",
    "W........................W",
    "W...........A............W",
    "WWWW.......O..O.......WWWW",
    "W........................W",
    "W......WWWW....WWWW......W",
    "W......W..W....W..W......W",
    "W......WO.W....W.OW......W",
    "W......WWWW....WWWW......W",
    "W........................W",
    "WWWW.......O..O.......WWWW",
    "W.....WWWW......WWWW.....W",
    "W.....W............W.....W",
    "W..O..W............W..E..W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWW"
]

def get_mall_map():
    return parse_ascii_map(MALL_ASCII)
