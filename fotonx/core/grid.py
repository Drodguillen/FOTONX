from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid2D:
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float

    @property
    def Nx(self) -> int:
        return int(self.x.size)

    @property
    def Ny(self) -> int:
        return int(self.y.size)

    @property
    def N(self) -> int:
        return int(self.Nx * self.Ny)


