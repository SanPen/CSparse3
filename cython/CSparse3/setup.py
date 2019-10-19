from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# python code to cython
# setup(ext_modules=cythonize('csparse3.py', language_level=3),)

# compile
ext_modules = [Extension("csparse3", ["csparse3.pyx"]),]
setup(name='csparse3',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
