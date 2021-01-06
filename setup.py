"""Setup script."""

import setuptools
import subprocess
import sys

# Install Gurobi interface. Use `pip -v` to see subprocess outputs.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-i', 'https://pypi.gurobi.com', 'gurobipy'])

setuptools.setup(
    name='cobmo',
    version='0.3.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        'cvxpy',
        'CoolProp==6.2.1',
        'hvplot',
        'kaleido',  # For static plot output with plotly.
        'matplotlib',
        'multimethod',
        'numpy',
        'pandas',
        'parameterized',  # for tests.
        'plotly',
        'pvlib',
        'scipy'
    ])
