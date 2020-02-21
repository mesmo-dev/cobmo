"""Installation script."""

import setuptools

setuptools.setup(
    name='cobmo',
    version='0.3.0',
    py_modules=setuptools.find_packages(),
    install_requires=[
        'CoolProp==6.2.1',
        'hvplot',
        'multimethod',
        'numpy',
        'pandas',
        'parameterized',  # for tests.
        'pvlib',
        'pyomo',
        'scipy',
        'seaborn'
    ])
