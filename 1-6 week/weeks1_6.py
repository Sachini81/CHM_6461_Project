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

    def valid(self) -> bool:
        """Convenience: both self-avoiding and connected."""
        return self.self_avoiding() and self.connected()

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
# WEEK 4: MONTE CARLO MOVES
# =========================
def rot90(v: Point, k: int) -> Point:
    """
    Rotate vector v around origin by k*90 degrees, k in {0,1,2,3}.
    """
    x, y = v
    if k % 4 == 0:
        return (x, y)
    if k % 4 == 1:
        return (-y, x)
    if k % 4 == 2:
        return (-x, -y)
    return (y, -x)


def try_pivot_move(coords: List[Point], rng: random.Random) -> Optional[List[Point]]:
    """
    Pivot move:
      choose pivot index i (not endpoints), rotate one side around pivot by 90/180/270 degrees.
    """
    n = len(coords)
    if n < 4:
        return None

    i = rng.randrange(1, n - 1)
    pivot = coords[i]
    k = rng.choice([1, 2, 3])  # 90, 180, 270
    rotate_tail = rng.random() < 0.5

    new_coords = list(coords)

    if rotate_tail:
        for j in range(i + 1, n):
            v = (coords[j][0] - pivot[0], coords[j][1] - pivot[1])
            rv = rot90(v, k)
            new_coords[j] = (pivot[0] + rv[0], pivot[1] + rv[1])
    else:
        for j in range(0, i):
            v = (coords[j][0] - pivot[0], coords[j][1] - pivot[1])
            rv = rot90(v, k)
            new_coords[j] = (pivot[0] + rv[0], pivot[1] + rv[1])

    if Polymer2D(new_coords).valid():
        return new_coords
    return None


def try_end_move(coords: List[Point], rng: random.Random) -> Optional[List[Point]]:
    """
    End move:
      move one end bead to a different free neighbor of its adjacent bead.
    """
    n = len(coords)
    if n < 2:
        return None

    new_coords = list(coords)
    move_head = rng.random() < 0.5

    if move_head:
        anchor = coords[1]
        occupied = set(coords[1:])  # all except head
        candidates = [p for p in Lattice2D.neighbors(anchor) if p not in occupied]
        if not candidates:
            return None
        new_coords[0] = rng.choice(candidates)
    else:
        anchor = coords[-2]
        occupied = set(coords[:-1])  # all except tail
        candidates = [p for p in Lattice2D.neighbors(anchor) if p not in occupied]
        if not candidates:
            return None
        new_coords[-1] = rng.choice(candidates)

    if Polymer2D(new_coords).valid():
        return new_coords
    return None


def try_crankshaft_move(coords: List[Point], rng: random.Random) -> Optional[List[Point]]:
    """
    Crankshaft move (simple lattice version):
      pick i (1..n-3), attempt to rotate the two middle beads within a 2x1 rectangle.
    """
    n = len(coords)
    if n < 4:
        return None

    i = rng.randrange(1, n - 2)
    a = coords[i - 1]
    b = coords[i]
    c = coords[i + 1]
    d = coords[i + 2]

    # endpoints must be 2 steps apart to form rectangle corners
    if manhattan(a, d) != 2:
        return None

    # corners that are neighbors of both a and d
    corners = [p for p in Lattice2D.neighbors(a) if manhattan(p, d) == 1]

    possible_pairs = []
    for pb in corners:
        for pc in corners:
            if pb == pc:
                continue
            if manhattan(pb, pc) != 1:
                continue
            # ensure a-pb-pc-d connectivity
            if manhattan(a, pb) == 1 and manhattan(pb, pc) == 1 and manhattan(pc, d) == 1:
                possible_pairs.append((pb, pc))

    if not possible_pairs:
        return None

    # remove current configuration if present
    possible_pairs = [pair for pair in possible_pairs if not (pair[0] == b and pair[1] == c)]
    if not possible_pairs:
        return None

    pb, pc = rng.choice(possible_pairs)
    new_coords = list(coords)
    new_coords[i] = pb
    new_coords[i + 1] = pc

    if Polymer2D(new_coords).valid():
        return new_coords
    return None


def propose_move(coords: List[Point], rng: random.Random) -> Optional[List[Point]]:
    """
    Mix of moves (tune weights if you want):
      50% pivot, 30% crankshaft, 20% end move
    """
    r = rng.random()
    if r < 0.50:
        return try_pivot_move(coords, rng)
    elif r < 0.80:
        return try_crankshaft_move(coords, rng)
    else:
        return try_end_move(coords, rng)


# =========================
# WEEK 5: HP MODEL ENERGY
# =========================
def hp_energy(coords: List[Point], seq: str, eps_hh: float = -1.0) -> float:
    """
    HP energy:
      each NON-bonded nearest-neighbor H-H contact contributes eps_hh (negative).

    Rules:
      - seq length must equal n
      - count each contact once
      - exclude bonded neighbors (i,i+1)
    """
    n = len(coords)
    if len(seq) != n:
        raise ValueError("Sequence length must match polymer length.")

    pos_to_index = {coords[i]: i for i in range(n)}
    E = 0.0

    for i in range(n):
        if seq[i] != "H":
            continue
        for nb in Lattice2D.neighbors(coords[i]):
            j = pos_to_index.get(nb, None)
            if j is None or j <= i:
                continue
            if seq[j] != "H":
                continue
            if abs(i - j) == 1:
                continue
            E += eps_hh

    return E


# =========================
# WEEK 6: METROPOLIS MC + TEMPERATURE SWEEP
# =========================
def metropolis_accept(dE: float, T: float, rng: random.Random) -> bool:
    """
    kB = 1 units.
    Accept if dE <= 0, else with probability exp(-dE/T).
    """
    if dE <= 0:
        return True
    if T <= 0:
        return False
    return rng.random() < math.exp(-dE / T)


def run_mc_hp(
    n: int,
    seq: str,
    T: float,
    n_steps: int,
    burn_in: int,
    sample_every: int,
    seed: int = 0,
) -> Dict[str, float]:
    """
    MC simulation at one temperature T.
    Returns: <E>, Cv, <R>, <Rg>, acceptance rate, number of samples.
    """
    rng = random.Random(seed)

    # start from a valid SAW
    coords = generate_saw(n=n, max_tries=8000, rng=rng)
    E = hp_energy(coords, seq)

    accepted = 0
    proposed = 0

    E_samples: List[float] = []
    R_samples: List[float] = []
    Rg_samples: List[float] = []

    for step in range(n_steps):
        new_coords = propose_move(coords, rng)
        if new_coords is None:
            continue  # invalid move proposal
        proposed += 1

        newE = hp_energy(new_coords, seq)
        dE = newE - E

        if metropolis_accept(dE, T, rng):
            coords = new_coords
            E = newE
            accepted += 1

        if step >= burn_in and ((step - burn_in) % sample_every == 0):
            E_samples.append(E)
            R_samples.append(end_to_end_distance(coords))
            Rg_samples.append(radius_of_gyration(coords))

    if not E_samples:
        raise RuntimeError("No samples collected. Reduce burn_in/sample_every or increase n_steps.")

    E_mean = sum(E_samples) / len(E_samples)
    R_mean = sum(R_samples) / len(R_samples)
    Rg_mean = sum(Rg_samples) / len(Rg_samples)

    # Cv from fluctuations: Cv = (<E^2> - <E>^2) / T^2  (kB=1)
    E2_mean = sum(e * e for e in E_samples) / len(E_samples)
    Cv = (E2_mean - E_mean * E_mean) / (T * T) if T > 0 else float("nan")

    acc_rate = accepted / proposed if proposed > 0 else 0.0

    return {
        "T": T,
        "E_mean": E_mean,
        "Cv": Cv,
        "R_mean": R_mean,
        "Rg_mean": Rg_mean,
        "acc_rate": acc_rate,
        "n_samples": float(len(E_samples)),
    }


# =========================
# OUTPUT SAVING
# =========================
def save_output(text: str, filename: str = "weeks1_6_output.txt") -> str:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


# =========================
# MAIN (Weeks 1–6)
# =========================
if __name__ == "__main__":
    lines: List[str] = []
    lines.append("LATTICE POLYMER MONTE CARLO (Weeks 1–6)")
    lines.append(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("-" * 72)

    # -----------------
    # Settings you can change
    # -----------------
    n = 20
    seq = "HPPHPPHPHPPHPPHPHPPH"  # must be length n

    temps = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    n_steps = 20000
    burn_in = 5000
    sample_every = 50

    lines.append(f"Polymer length n = {n}")
    lines.append(f"HP sequence     = {seq}")
    lines.append("")

    # Week 1 demo
    lattice = Lattice2D()
    coords1 = straight_chain(6)
    for p in coords1:
        lattice.occupy(p)
    polymer1 = Polymer2D(coords1)
    lines.append("WEEK 1 DEMO: Straight chain summary")
    lines.append(str(polymer1.summary()))
    lines.append("")

    # Week 2 + Week 3 demo: one SAW
    rng_demo = random.Random(1)
    coords2 = generate_saw(n=n, max_tries=8000, rng=rng_demo)
    polymer2 = Polymer2D(coords2)
    lines.append("WEEK 2/3 DEMO: One SAW + observables")
    lines.append(f"Valid SAW? {polymer2.valid()}")
    lines.append(f"R  = {end_to_end_distance(coords2):.6f}")
    lines.append(f"Rg = {radius_of_gyration(coords2):.6f}")
    lines.append(f"HP energy E = {hp_energy(coords2, seq):.6f}  (eps_HH = -1)")
    lines.append("")

    # Week 3 demo: ensemble averages of SAW-only (no energy)
    n_samples = 2000
    stats = sample_saws(n=n, n_samples=n_samples, seed=2, max_tries=8000)
    lines.append("WEEK 3 DEMO: Ensemble averages over many SAWs (growth sampling)")
    lines.append(f"n = {stats['n']}   samples = {stats['n_samples']}")
    lines.append(f"<R>  = {stats['R_mean']:.6f} ± {stats['R_std']:.6f}")
    lines.append(f"<Rg> = {stats['Rg_mean']:.6f} ± {stats['Rg_std']:.6f}")
    lines.append("")

    # Weeks 4–6: MC + temperature sweep
    lines.append("WEEK 4–6: Metropolis MC with mixed moves (pivot/crankshaft/end)")
    lines.append(f"MC steps      = {n_steps}")
    lines.append(f"Burn-in steps = {burn_in}")
    lines.append(f"Sample every  = {sample_every} steps")
    lines.append("")
    lines.append("Results (kB=1 units):")
    lines.append("T\t<E>\t\tCv\t\t<R>\t\t<Rg>\t\tacc_rate\tn_samples")

    for k, T in enumerate(temps):
        out = run_mc_hp(
            n=n,
            seq=seq,
            T=T,
            n_steps=n_steps,
            burn_in=burn_in,
            sample_every=sample_every,
            seed=100 + k,
        )
        lines.append(
            f"{out['T']:.2f}\t"
            f"{out['E_mean']:.6f}\t"
            f"{out['Cv']:.6f}\t"
            f"{out['R_mean']:.6f}\t"
            f"{out['Rg_mean']:.6f}\t"
            f"{out['acc_rate']:.4f}\t"
            f"{int(out['n_samples'])}"
        )

    lines.append("")
    lines.append("NOTE:")
    lines.append("- Moves: pivot, crankshaft, end moves; invalid proposals are skipped.")
    lines.append("- HP energy: counts NON-bonded nearest-neighbor H-H contacts (each contributes -1).")
    lines.append("- Cv is estimated from fluctuations: Cv = (<E^2> - <E>^2) / T^2 (kB=1).")
    lines.append("")

    out_text = "\n".join(lines)
    print(out_text)

    out_file = save_output(out_text, filename="weeks1_6_output.txt")
    print(f"\nSaved output to: {out_file}")
