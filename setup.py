"""Setup script."""

import setuptools

setuptools.setup(
    name='cobmo',
    version='0.3.0',
    packages=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        'cvxpy',
        'gurobipy',
        'hvplot',
        'kaleido',  # For static plot output with plotly.
        'matplotlib',
        'multimethod',
        'numpy',
        'pandas',
        'plotly',
        'psychrolib',
        'pvlib',
        'scipy',
        'tqdm',
    ],
    extras_require={"tests": [
        "flake8",
        "parameterized",
        "pytest",
        "pytest-cov",
        "pytest-subtests",
    ]}
)
