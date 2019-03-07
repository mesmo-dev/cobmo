"""
Building model installation script
"""

from setuptools import setup, find_packages

setup(
    name='cobmo',
    version='0.1',
    py_modules=find_packages(),
    install_requires=[
        'CoolProp',
        'numpy',
        'pandas',
        'pvlib',
        'pyomo',
        'scipy'
    ])
