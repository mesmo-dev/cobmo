# CoBMo - Control-oriented Building Model

[![DOI](https://zenodo.org/badge/173782015.svg)](https://zenodo.org/badge/latestdoi/173782015)

The Control-oriented Building Model (CoBMo) is a building modelling framework catering specifically for the formulation of MPC problems for thermal building systems by keeping all model equations in the linear, i.e., convex, domain. CoBMo provides a mathematical model which expresses the relationship between the electric load of the thermal building systems and the indoor air climate with consideration for interactions of the building with its environment, its occupants and appliances. To this end, CoBMo currently implements models for 1) the thermal comfort of building occupants as well as 2) the indoor air quality.

## Work in Progress

Please note that the repository is under active development and the interface may change without notice. Create an [issue](https://github.com/TUMCREATE-ESTL/cobmo/issues) if you have ideas / comments / criticism that may help to make the tool more useful.

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

## Acknowledgements

- Sebastian Troitzsch developed and implemented the initial version of CoBMo and maintains this repository.
- Sarmad Hanif and Henryk Wolisz initiated and supervised the work on the initial version of CoBMo.
- Anthony Vautrin under supervision of Srikkanth Ramachandran developed the models for indoor air quality and demand controlled ventilation.
- Tommaso Miori developed the models for thermal and battery energy storage.
- Sherif Hashem under supervision of Thomas Licklederer developed the hydronic radiator models.
- This work was financially supported by the Singapore National Research Foundation under its Campus for Research Excellence And Technological Enterprise (CREATE) programme.
