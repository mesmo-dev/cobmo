"""Setup script."""

import setuptools

setuptools.setup(
    name='cobmo',
    version='0.3.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
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
        'pyomo',
        'scipy',
        'seaborn'
    ])
