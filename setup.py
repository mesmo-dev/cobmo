"""
Building model installation script
"""

from setuptools import setup, find_packages

setup(
    name='cobmo',
    version='0.1',
    py_modules=find_packages(),
    install_requires=[
        'CoolProp==6.2.1',
        'hvplot',
        'numpy',
        'pandas',
        'pvlib',
        'pyomo',
        'scipy',
        'seaborn'
    ])
