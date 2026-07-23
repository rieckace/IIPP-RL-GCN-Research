from environment.grid import Grid
from environment.constants import CellType

grid = Grid(10, 10)

grid.set_cell(1, 1, CellType.AGENT)
grid.set_cell(8, 8, CellType.EXIT)
grid.set_cell(5, 5, CellType.SMOKE)

grid.display()