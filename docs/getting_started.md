# Getting started

## Installation

### Quick installation

1. Check requirements:
    - Python 3.7
    - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository.
3. In your Python environment, run:
    1. `pip install -v -e path_to_repository`

### Recommended installation

The following installation procedure requires additional steps, but can improve performance and includes optional dependencies. For example, the numpy, pandas and cvxpy packages are installed through Anaconda, which ensures the use of more performant math libraries.

1. Check requirements:
    - [Anaconda Python Distribution](https://www.anaconda.com/distribution/)
    - [Gurobi Optimizer](http://www.gurobi.com/) or [CPLEX Optimizer](https://www.ibm.com/analytics/cplex-optimizer)
2. Clone or download repository.
3. In Anaconda Prompt, run:
    1. `conda create -n cobmo python=3.7`
    2. `conda activate cobmo`
    3. `conda install -c conda-forge cvxpy numpy pandas`
    4. `pip install -v -e path_to_repository`
4. If you want to use CPLEX:
    1. Install CPLEX Python interface (see latest CPLEX documentation).
    2. Create or modify `config.yml` (see below in "Configuration with `config.yml`").

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

The `examples` directory contains run scripts which demonstrate the usage of CoBMo.

- `run_example.py`: Example run script for using the building model.
- `run_storage_planning_example.py`: Run script for single simulation / optimization of sensible thermal or battery storage.
- `run_storage_planning_battery_cases.py`: Run script for BES cases lifetime.
- `run_validation.py`: Run script for building model validation.
- `run_evaluation_load_reduction.py`: Run script for evaluating demand side flexibility in terms of load reduction.
- `run_evaluation_price_sensitivity.py`: Run script for evaluating demand side flexibility in terms of price sensitivity.
- (Further example scripts for development / testing of new features may not yet be documented here.)

## Configuration with `config.yml`

CoBMo configuration parameters (e.g. the output format of plots) can be set in `config.yml`. As an initial user, you most likely will not need to modify the configuration.

If you want to change the configuration, you can create or modify `config.yml` in the CoBMo repository main directory. CoBMo will automatically create `config.yml` if it does not exist. Initially, `config.yml` will be empty. You can copy configuration parameters from `cobmo/config_default.yml` to `config.yml` and modify their value to define your local configuration. To define nested configuration parameters, you need to replicate the nested structure in `config.yml`. For example, to define CPLEX as the optimization solver, use:

```
optimization:
  solver_name: cplex
```

The configuration parameters which are defined in `config.yml` will take precedence over those defined in `cobmo/config_default.yml`. If you would like to revert a parameter to its default value, just delete the parameter from `config.yml`. Please do not modify `cobmo/config_default.yml` directly.

## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).
