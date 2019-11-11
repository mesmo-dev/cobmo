# Getting Started

## Installation

1. Check requirements:
    - Python 3.7
    - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository.
3. In your Python environment, run `pip install -e path_to_cobmo_repository`.

### Alternative Installation

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

## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).
