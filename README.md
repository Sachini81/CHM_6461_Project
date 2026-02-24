# Lattice Polymer Monte Carlo Project (Weeks 1–3)
**Model:** 2D square lattice  
**Focus :** Self-Avoiding Walk (SAW) representation + SAW generation + basic polymer observables

This repository contains a single Python script that implements:
- **Week 1:** Lattice + Polymer data structures (occupancy, connectivity, self-avoidance)
- **Week 2:** Random-growth **SAW generator** (with restarts if trapped)
- **Week 3:** Polymer observables (**end-to-end distance R**, **radius of gyration Rg**) and ensemble statistics over many SAWs

> Note: The SAW generator uses a **simple growth + restart** strategy. This is good for initial validation and Week 2/3 tasks. A true Monte Carlo move set (pivot/crankshaft/etc.) is typically added later for proper equilibrium sampling.

---

## Files
- `lattice_representation.py`  
  Main script for Weeks 1–3. Runs demos and saves output to a text file.
- `weeks1_3_output.txt`  
  Output file automatically created after running the script.

---

## Requirements
- Python **3.14.3** 







