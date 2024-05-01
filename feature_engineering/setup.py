from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("featmap_gen_cython", ["featmap_gen_cython.pyx"], include_dirs=[np.get_include()])
]

setup(
    ext_modules=cythonize(extensions)
)