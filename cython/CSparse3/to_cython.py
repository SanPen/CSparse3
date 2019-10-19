import os
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('csparse3.py', language_level=3),)
