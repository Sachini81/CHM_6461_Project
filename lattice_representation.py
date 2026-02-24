from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict
import random
import math
from datetime import datetime

Point = Tuple[int, int]


# =========================
# WEEK 1: LATTICE + POLYMER
# =========================
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
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


@dataclass
class Polymer2D:
    coords: List[Point]
    bonds: List[Tuple[int, int]]

    def __init__(self, coords: List[Point]):
        self.coords = list(coords)
        self.bonds = [(i, i + 1) for i in range(len(self.coords) - 1)]

    def self_avoiding(self) -> bool:
        return len(set(self.coords)) == len(self.coords)

    def connected(self) -> bool:
        return all(manhattan(self.coords[i], self.coords[i + 1]) == 1
                   for i in range(len(self.coords) - 1))

    def summary(self) -> dict:
        return {
            "n_beads": len(self.coords),
            "coords": self.coords,
            "bonds": self.bonds,
            "self_avoiding": self.self_avoiding(),
            "connected": self.connected(),
        }


def straight_chain(n: int) -> List[Point]:
    return [(i, 0) for i in range(n)]


# =========================
# WEEK 2: SAW GENERATION
# =========================
def generate_saw(
    n: int,
    max_tries: int = 3000,
    rng: Optional[random.Random] = None,
) -> List[Point]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return [(0, 0)]

    rng = rng or random.Random()

    for _attempt in range(max_tries):
        lattice = Lattice2D()
        coords: List[Point] = [(0, 0)]
        lattice.occupy((0, 0))

        trapped = False
        for _i in range(1, n):
            cur = coords[-1]
            candidates = [p for p in lattice.neighbors(cur) if lattice.is_free(p)]
            if not candidates:
                trapped = True
                break
            nxt = rng.choice(candidates)
            coords.append(nxt)
            lattice.occupy(nxt)

        if not trapped:
            return coords

    raise RuntimeError(
        f"Failed to generate SAW of length n={n} after {max_tries} restarts. "
        "Increase max_tries or reduce n."
    )


# =========================
# WEEK 3: OBSERVABLES + STATS
# =========================
def end_to_end_distance(coords: List[Point]) -> float:
    (x0, y0) = coords[0]
    (x1, y1) = coords[-1]
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def radius_of_gyration(coords: List[Point]) -> float:
    n = len(coords)
    x_cm = sum(p[0] for p in coords) / n
    y_cm = sum(p[1] for p in coords) / n
    rg2 = sum((p[0] - x_cm) ** 2 + (p[1] - y_cm) ** 2 for p in coords) / n
    return math.sqrt(rg2)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        raise ValueError("Empty list.")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def sample_saws(
    n: int,
    n_samples: int,
    seed: int = 0,
    max_tries: int = 5000,
) -> Dict[str, object]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = random.Random(seed)

    Rs: List[float] = []
    Rgs: List[float] = []

    for _ in range(n_samples):
        coords = generate_saw(n=n, max_tries=max_tries, rng=rng)
        poly = Polymer2D(coords)
        assert poly.self_avoiding()
        assert poly.connected()

        Rs.append(end_to_end_distance(coords))
        Rgs.append(radius_of_gyration(coords))

    R_mean, R_std = mean_std(Rs)
    Rg_mean, Rg_std = mean_std(Rgs)

    return {
        "n": n,
        "n_samples": n_samples,
        "R_mean": R_mean,
        "R_std": R_std,
        "Rg_mean": Rg_mean,
        "Rg_std": Rg_std,
        "R_values": Rs,
        "Rg_values": Rgs,
    }


# =========================
# OUTPUT SAVING
# =========================
def save_output(text: str, filename: str = "weeks1_3_output.txt") -> str:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


# =========================
# MAIN (Weeks 1–3)
# =========================
if __name__ == "__main__":
    lines: List[str] = []
    lines.append("LATTICE POLYMER PROJECT (Weeks 1–3)")
    lines.append(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("-" * 60)

    # Week 1
    lattice = Lattice2D()
    coords1 = straight_chain(6)
    for p in coords1:
        lattice.occupy(p)
    polymer1 = Polymer2D(coords1)
    lines.append("WEEK 1 DEMO: Straight chain summary")
    lines.append(str(polymer1.summary()))
    lines.append("")

    # Week 2
    n = 25
    rng = random.Random(1)
    coords2 = generate_saw(n=n, max_tries=8000, rng=rng)
    polymer2 = Polymer2D(coords2)
    lines.append("WEEK 2 DEMO: One SAW summary")
    lines.append(str(polymer2.summary()))
    lines.append(f"End-to-end distance R = {end_to_end_distance(coords2):.6f}")
    lines.append(f"Radius of gyration Rg = {radius_of_gyration(coords2):.6f}")
    lines.append("")

    # Week 3
    n_samples = 2000
    stats = sample_saws(n=n, n_samples=n_samples, seed=2, max_tries=8000)
    lines.append("WEEK 3 DEMO: Ensemble averages over many SAWs")
    lines.append(f"n = {stats['n']}   samples = {stats['n_samples']}")
    lines.append(f"<R>  = {stats['R_mean']:.6f} ± {stats['R_std']:.6f}")
    lines.append(f"<Rg> = {stats['Rg_mean']:.6f} ± {stats['Rg_std']:.6f}")
    lines.append("")

    out_text = "\n".join(lines)
    print(out_text)

    out_file = save_output(out_text, filename="weeks1_3_output.txt")
    print(f"\nSaved output to: {out_file}")