# CoBMo - Control-oriented Building Model

[![DOI](https://zenodo.org/badge/173782015.svg)](https://zenodo.org/badge/latestdoi/173782015)

The Control-oriented Building Model (CoBMo) is a building modelling framework catering specifically for the formulation of MPC problems for thermal building systems by keeping all model equations in the linear, i.e., convex, domain. CoBMo provides a mathematical model which expresses the relationship between the electric load of the thermal building systems and the indoor air climate with consideration for interactions of the building with its environment, its occupants and appliances. To this end, CoBMo currently implements models for 1) the thermal comfort of building occupants as well as 2) the indoor air quality.

## Documentation

The preliminary CoBMo documentation is located at: [cobmo.readthedocs.io](https://cobmo.readthedocs.io/)

## Installation

1. Check requirements:
    - Python 3.7
    - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository.
3. In your Python environment, run `pip install -e path_to_cobmo_repository`.

Please also read [docs/intro.md](./docs/intro.md).

## Contributing

If you are keen to contribute to this project, please see [docs/contributing.md](./docs/contributing.md).
