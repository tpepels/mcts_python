from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy

ext_modules = [
    Extension(
        "games.amazons",
        ["games/amazons.py"],
        include_dirs=[numpy.get_include()],  # Include directory for numpy
    ),
    Extension(
        "games.tictactoe",
        ["games/tictactoe.py"],
        include_dirs=[numpy.get_include()],  # Include directory for numpy
    ),
    Extension(
        "games.breakthrough",
        ["games/breakthrough.py"],
        include_dirs=[numpy.get_include()],  # Include directory for numpy
    ),
    Extension("ai.alpha_beta", ["ai/alpha_beta.pyx"]),
    Extension("ai.transpos_table", ["ai/transpos_table.py"], include_dirs=[numpy.get_include()]),
]

setup(name="Cython MCTS", ext_modules=cythonize(ext_modules))
