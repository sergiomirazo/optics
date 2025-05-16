# FresnelOptics

### Overview

**FresnelOptics** is a Python-based library for calculating various optical properties of dielectric interfaces. It includes classes to model simple interfaces, material slabs (etalons), and a transfer matrix module for analyzing light transmission and reflection through a stack of dielectric layers.

### Key Features

- **Interfaces**: Model dielectric interfaces using the `Interface` class.
- **Material Slabs**: Analyze transmission and reflection of dielectric slabs with the `Plate` class.
- **Transfer Matrix**: Calculate transmission and reflection for complex layer stacks using the transfer matrix method.
- **Uniaxial Layers**: Handle emission and reflection in uniaxial layers, specifically where the optical axis is perpendicular to the slab plane, useful in semiconductor and quantum well structures.

### Requirements

- `numpy`
- `matplotlib`

### Unique Capabilities

Unlike many existing thin films codes, FresnelOptics can analyze uniaxial layers under specific conditions, making it valuable for certain semiconductor structures and intersubband transitions of quantum wells.

### Flexibility

The library takes advantage of Python's flexibility, ensuring easy adaptation for those comfortable with the language. The code prioritizes readability over rigorous input validation.

### Example Usage

Each module comes with example code demonstrating their use. Additional examples can be found in the `examples` directory. These examples use `init.py` to locate FresnelOptics, which eliminates the need for installation into the `site-packages` directory.

---

## Module Summary

### fesnel.py

- **Class**: `Interface`
  - **Usage**: `Interface(n1=1.0, n2=1.0, theta=0.0)`
  - Models dielectric interfaces.

### materials.py

- **Class**: `Material`
  - Provides classes for calculating different dielectric constants.
  - Natural and real frequencies may be used; check comments for model specifics.

### optical_plate.py

- **Class**: `Plate`
  - **Usage**: `Plate(n1, d, w, theta, n0=1.0, n2=1.0)`
  - Models slab transmission/reflection.

### effective_medium.py

- **Classes**: 
  - `EffectiveMedium`: Calculate effective dielectric constants for thin layer stacks.
  - `EffectiveMedium_eps`: Similar but uses dielectric constants.

### uniaxial_plate.py

- **Classes**: 
  - `AnisoInterface`
  - `AnisoPlate`
  - Models uniaxial medium interfaces/slabs under specific optical conditions.

### uniaxial_plate2.py

- **Class**: 
  - `AnisoPlate`
  - A variation with a different derivation.

### transfer_matrix.py

- **Classes**: 
  - `Layer`, `LayerUniaxial`, `Filter`
  - Further details in the module.

### layer_types.py

- Adjusts material types for compatibility with the transfer matrix.

### incoherent_transfer_matrix.py

- **Class**: `IncoherentFilter`
  - Combines coherent and incoherent filters.

---

By leveraging FresnelOptics's capabilities, researchers in optics can simulate complex dielectric structures, analyze optical phenomena, and explore new insights into uniaxial and layered materials.
