"""
A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# but we need distutils because of the numba integration...
from distutils.core import setup

import os
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

from CSparse3.__version__ import CSparse3Version
from CSparse3.float_numba import cc


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
if os.path.exists(os.path.join(here, '..', 'README.md')):
    with open(os.path.join(here, '..', 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ''

if os.path.exists(os.path.join(here, '..', 'doc', 'about.rst')):
    with open(os.path.join(here, '..', 'doc', 'about.rst'), encoding='utf-8') as f:
        description = f.read()
else:
    description = ''


base_path = os.path.join('CSparse3')

packages = find_packages(exclude=['docs', 'test', 'research', 'tests'])

package_data = {}

dependencies = ["numba>=0.46",
                "numpy>=1.14.0",
                ]

setup(name = "CSparse3",
      version = CSparse3Version,
      description = "Sparse matrix library",
      author = "Santiago Peñate-Vera",
      author_email = "santiago.penate.vera@gmail.com",
      url = "https://github.com/SanPen/CSparse3",
      #Name the folder where your packages live:
      #(If you have other packages (dirs) or modules (py files) then
      #put them into the package directory - they will be found
      #recursively.)
      packages = packages,
      #'package' package must contain files (see list above)
      #I called the package 'package' thus cleverly confusing the whole issue...
      #This dict maps the package name =to=> directories
      #It says, package *needs* these files.
      package_data = package_data,
      #'runner' is in the root.
      scripts = [],
      long_description = """Sparse matrix library.""",
      #
      #This next part it for the Cheese Shop, look a little down the page.
      #classifiers = []
      ext_modules=[cc.distutils_extension()]
      )