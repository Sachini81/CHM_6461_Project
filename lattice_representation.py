

from dataclasses import dataclass
from typing import List, Tuple, Set

Point = Tuple[int, int]

def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

@dataclass
class Lattice2D:
    occupied: Set[Point]

    def __init__(self):
        self.occupied = set()

    def is_free(self, p: Point) -> bool:
        return p not in self.occupied

    def occupy(self, p: Point) -> None:
        self.occupied.add(p)

    def release(self, p: Point) -> None:
        self.occupied.remove(p)

    @staticmethod
    def neighbors(p: Point) -> List[Point]:
        x, y = p
        return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

@dataclass
class Polymer2D:
    coords: List[Point]  # monomer coordinates
    bonds: List[Tuple[int, int]]  # (i, i+1)

    def __init__(self, coords: List[Point]):
        self.coords = list(coords)
        self.bonds = [(i, i+1) for i in range(len(self.coords)-1)]

    def self_avoiding(self) -> bool:
        return len(set(self.coords)) == len(self.coords)

    def connected(self) -> bool:
        return all(manhattan(self.coords[i], self.coords[i+1]) == 1
                   for i in range(len(self.coords)-1))

    def summary(self) -> dict:
        return {
            "n_beads": len(self.coords),
            "coords": self.coords,
            "bonds": self.bonds,
            "self_avoiding": self.self_avoiding(),
            "connected": self.connected(),
        }

def straight_chain(n: int) -> List[Point]:
    return [(i, 0) for i in range(n)]  # simple 2D initial conformation

if __name__ == "__main__":
    lattice = Lattice2D()
    coords = straight_chain(6)
    for p in coords:
        lattice.occupy(p)

    polymer = Polymer2D(coords)
    print(polymer.summary())
