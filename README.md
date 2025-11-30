# ToyGl: Computational Biomechanics Simulation Framework

ToyGl is a Python codebase for simulating the mechanics and growth of three-dimensional epithelial cells modeled as deformable, triangulated surfaces. Each cell is represented as a network of mass points (nodes) connected by elastic elements (filaments) and organized into triangular facets; collections of cells form an epithelium. The model combines a mass–spring description of the cell cortex with an internal pressure–volume relation and optional inter-cell adhesions.

The current implementation focuses on:

- Initializing one or several cells from an icosahedral (or refined icosphere) discretization.
- Evolving nodal positions in time under the action of elastic, pressure, gravitational, viscous, and contact forces.
- Allowing cell growth by adjusting target volumes and, in legacy variants, refining the surface mesh.
- Exporting cell surfaces at each time step for analysis and 3D visualization (e.g. in ParaView or PyVista).

The main entry point is `run_simulation.py`, which constructs an `Epithelium` object, advances the system using explicit time integration, and writes out the evolving cell geometry. The core numerical and geometric logic is implemented in the `src/` directory (nodes, filaments, facets, cells, and epithelial tissue), while the `paraview_macros/` and plotting utilities support post-processing and figure generation.

## How to run

### 1. Install environment

ToyGl is designed to run with Python 3.10 and a small set of scientific Python libraries. A compatible Conda environment is described in `environment.yml`:

- Python 3.10
- `numpy`
- `scipy`
- `pyvista`
- `natsort`
- `dill`

To create and activate the environment with Conda or Mamba:

```bash
cd /home/ardati/PycharmProjects/toygl
conda env create -f environment.yml
conda activate ToyGl
```

(or use `mamba env create -f environment.yml` if you prefer Mamba.)

### 2. Run a basic simulation

The main entry point for simulations is `run_simulation.py`, which:

- Builds an epithelial tissue (`Epithelium`) with one (or more) 3D cells.
- Evolves the system in time using explicit time stepping.
- Writes per-step facet geometry to disk for visualization.

From the project root:

```bash
python run_simulation.py
```

By default this will:

- Initialize a single growing cell (`Epithelium.create_an_eptm_of_a_growing_cells()`).
- Run the dynamic evolution loop up to the configured number of steps (`evolution_limit`).
- Export facet geometry for each step into the `Data_ToyGL/fastoutputfacets/` directory (as configured in `run_simulation.py`).
- Optionally measure and append total runtime to `simulation_times.txt`.

Some behavior is controlled in `run_simulation.py` via simple flags and constants:

- `evolution_limit`: maximum number of time steps.
- `dt`: time step used by the integrator.
- `Recup`: if `True`, resume from a previously pickled tissue instead of initializing a new one.
- `save_steps`: if `True`, pickle the `Epithelium` object at each step to disk.

### 3. Visualize results

Facet geometry exported by `Epithelium.fast_export_facets` can be visualized with:

- ParaView (using the provided `paraview_macros/` scripts), or
- Custom Python tools in `src/plotting_utils.py` / PyVista.

Each exported file encodes the cell surface at a given time step and can be used to reconstruct sequences like:

- The initial icosahedral cell (`ToyGl_cell_icosahedre.png`).
- The highly subdivided, grown cell with thousands of facets (`Cell_at_3000_facet.png`).

## Core model objects

ToyGl’s mechanical model is built from five core classes defined in `src/`:

- **`Node` (`src/node_class.py`)**  
  Represents a material point of the cell surface or interior skeleton.
  - State: position, velocity, accumulated forces (including `pressure_forces`).
  - Physical properties: mass, radius, stiffness, color.
  - Forces: ground contact, gravity, viscous damping; supports generic `add_force` and `reset_forces`.
  - Utility: can be blocked (fixed) to model boundary conditions.

- **`Filament` (`src/filament_class.py`)**  
  Represents a mechanical connection (spring) between two nodes.
  - Connects two `Node` objects and belongs to a `Cell` (or acts as an inter-cell adhesion).
  - Parameters: rest length (`lfree`), axial stiffness (`rig_EA`), radius, color.
  - Computes its elastic force along the current filament vector and applies equal and opposite forces on its end nodes.
  - Tracks age in time steps (useful for growth/remodeling rules and adhesions).

- **`Facet` (`src/facet_class.py`)**  
  Represents a triangular surface element of a cell.
  - Defined by three `Filament` edges and, implicitly, three `Node` vertices.
  - Computes and stores an outward unit normal.
  - Used to discretize the closed cell surface, compute surface area, and distribute pressure forces to nodes.

- **`Cell` (`src/cell_class.py`)**  
  Represents a single 3D cell as a closed triangulated shell.
  - Geometry: initialised as an icosahedron (or refined icosphere); maintains lists of `Node`, `Filament` and `Facet` objects.
  - Volume: computed from the convex hull / triangulated surface; stored as `volume`, with a reference `volume0` and a target volume.
  - Mechanics: internal pressure derived from a pressure–volume law; pressure forces applied to facet nodes along facet normals.
  - Growth: `grow_to_target_volume()` and related logic implement volume growth and (in legacy versions) filament/facet division rules.
  - Utility: translation in space, accessors `get_nodes()` and `get_filaments()` for use by the tissue-level model.

- **`Epithelium` (`src/eptm_class.py`)**  
  Represents a collection of one or more `Cell` objects forming an epithelial tissue.
  - Holds the list of cells, and provides a global view of all nodes, filaments, and facets.
  - Defines the global dynamical system state `state = [positions, velocities]` for all nodes across all cells.
  - Computes the time derivative of the state in `derive_state` by:
    - Resetting node forces and adding ground contact, gravity, and viscous damping.
    - Adding filament elastic forces (intra-cell plus optional adhesion filaments between cells).
    - Computing cell pressures and applying pressure forces from each `Cell`.
  - Advances the state in time via explicit integration (Euler or RK4) in `dynamic_evolution`.
  - Provides helper methods to construct standard initial tissues (one, two, five, or nine cells) and to export geometry for visualization.

## Legacy / prototypes

The repository also contains a set of historical prototype scripts that were used during the early development of ToyGl, mainly for OpenGL-based visualization and mesh experimentation. These files are **not** part of the current simulation pipeline driven by `run_simulation.py` and the `src/` package, but are kept for reference:

- `legacy/pyToyGL.py` – Original mass–spring + OpenGL prototype for a single 3D cell.
- `legacy/2pyToyGL.py` – Extended icosphere-based prototype with advanced logging and visualization.
- `legacy/frpyToyGL.py` – French-language variant of the original OpenGL prototype.
- `legacy/icosahedre.py` – Standalone icosphere/icosahedron visualization helper.
- `legacy/depot.py` – Experimental export and adhesion/neighbor utilities, not wired into the main pipeline.

These scripts can be useful to understand the historical evolution of the model or to explore alternative visualization ideas, but new development should target the main API in `src/` and `run_simulation.py`.
