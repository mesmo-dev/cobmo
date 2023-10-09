# Getting started

## Installation

1. Check requirements:
    - Python 3.7
    - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository.
3. In your Python environment, run `pip install -e path_to_cobmo_repository`.

### Alternative installation

If you are running into errors when installing or running CoBMo, this may be due to incompatibility with new versions of package dependencies, which have yet to be discovered and fixed. As a workaround, try installing CoBMo in an tested Anaconda environment via the the provided `environment.yml`, which represents the latest Anaconda Python environment in which CoBMo was tested and is expected to work.

1. Check requirements:
    - Windows 10
    - [Anaconda Distribution](https://www.anaconda.com/distribution/) (Python 3.x version)
    - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository.
4. In Anaconda Prompt, run `conda env create -f path_to_cobmo_repository/environment.yml`
5. Once the environment setup finished, run `conda activate cobmo` and `pip install -e path_to_cobmo_repository`.

``` important::
    Please also create an issue on Github if you run into problems with the normal installation procedure.
```

## Examples

The `examples` directory contains several run scripts which demonstrate possible usages of CoBMo:

- `run_example.py`: Example run script for using the building model.
- `run_storage_planning_example.py`: Run script for single simulation / optimization of sensible thermal or battery storage.
- `run_storage_planning_battery_cases.py`: Run script for BES cases lifetime.
- `run_validation.py`: Run script for building model validation.
- `run_evaluation_load_reduction.py`: Run script for evaluating demand side flexibility in terms of load reduction.
- `run_evaluation_price_sensitivity.py`: Run script for evaluating demand side flexibility in terms of price sensitivity.
- (Further example scripts for development / testing of new features may not yet be documented here.)

## Papers

The following papers have been prepared with CoBMo:

- [Preprint] Troitzsch, S., & Hamacher, T., **Control-oriented Thermal Building Modelling**, in IEEE PES General Meeting, Montreal, Canada, 2020. [`doi:10.36227/techrxiv.11923587`](https://doi.org/10.36227/techrxiv.11923587)
    - CoBMo [version 0.3.0](https://github.com/TUMCREATE-ESTL/cobmo/releases/tag/0.3.0) was used to prepare the results for this paper.
    - The related scripts are [`examples/run_evaluation_load_reduction.py`](https://github.com/TUMCREATE-ESTL/cobmo/blob/0.3.0/examples/run_evaluation_load_reduction.py) and [`examples/run_evaluation_price_sensitivity.py`](https://github.com/TUMCREATE-ESTL/cobmo/blob/0.3.0/examples/run_evaluation_price_sensitivity.py).
- Vautrin, A., Troitzsch, S., Ramachandran, S., & Hamacher, T., **Demand Controlled Ventilation for Electric Demand Side Flexibility**, in IBPSA Building Simulation Conference, Rome, Italy, 2019. [`doi:10.26868/25222708.2019.210968`](https://doi.org/10.26868/25222708.2019.210968)
    - A preliminary implementation of CoBMo was used to prepare the results for this paper.
    - The related scripts are currently not included in the repository.

## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).
