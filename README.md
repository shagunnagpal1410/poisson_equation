# 🔷 Meshfree 2D Poisson Equation Solvers (Dirichlet, Neumann, Mixed) — C++11

This repository provides meshfree solvers for the **2D Poisson equation** using the **Finite Point Method (FPM)**. Three separate implementations are provided based on the type of boundary condition:

- 🔴 `dirichlet.cpp` — Solves with **Dirichlet boundary conditions** on all boundaries.
- 🔵 `neumann.cpp` — Solves with **Neumann boundary conditions** on all boundaries.
- 🟡 `mixed.cpp` — Solves with a **combination of Dirichlet and Neumann boundary conditions**.

All implementations use **Gaussian-weighted least squares** for approximating differential operators and are built on **C++11** with the **Eigen** library for sparse linear algebra.

---

## 🧠 Method Overview

- **Discretization**: Meshfree point cloud on a unit square domain.
- **Laplacian Approximation**: Second-order polynomial basis.
- **Neighbor Search**: Voxel-based for spatial locality.
- **Weight Function**: Gaussian kernel.
- **Linear Solver**: `Eigen::SparseLU`.
- **Output**: CSV files containing temperature field and series-based analytical solution for comparison.

---

## 📁 Files in the Repository

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `dirichlet.cpp`       | FPM solver with Dirichlet boundary conditions (all sides).                  |
| `neumann.cpp`         | FPM solver with Neumann boundary conditions (all sides).                    |
| `mixed.cpp`           | FPM solver with mixed boundary conditions (Dirichlet + Neumann).            |
| `task.json`           | VS Code task runner config (enables Ctrl+Shift+B to build & run).           |
| `Eigen/`              | Eigen library directory (header-only).                                      |
| `Temperature.csv`     | Output from Dirichlet solver.                                               |
| `Temperature1.csv`    | Output from Neumann solver.                                                 |
| `Temperature2.csv`    | Output from Mixed BC solver.                                                |
| `PoissonSeriesSolution.csv` | Analytical solution for comparison (Dirichlet).                     |
| `PoissonSeriesSolution1.csv`| Analytical solution (Neumann).                                      |
| `PoissonSeriesSolution2.csv`| Analytical solution (Mixed).                                        |

---

## 🧰 Requirements

- C++11 compiler (e.g. `g++`, `clang++`)
- [Eigen](https://eigen.tuxfamily.org/) (already included in the repo or header path)
- Visual Studio Code (for easy `Ctrl+Shift+B` builds)

---

## ⚙️ How to Compile & Run

### 🖥 Using VS Code:
1. Open this folder in VS Code.
2. open task.json file and there you need to change 6june.cpp to which ever equation you want to solve say dirichlet.cpp do same for 6june.exe
3. Press **`Ctrl + Shift + B`** to build and run the selected `.cpp` file.
4. Make sure `task.json` points to the desired source file (`dirichlet.cpp`, etc).

### 📦 Compile Manually (if not using VS Code):
```bash
g++ -std=c++11 dirichlet.cpp -I ./Eigen -o dirichlet
./dirichlet
