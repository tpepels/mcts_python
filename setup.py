from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy

ext_modules = [
    Extension("games.gamestate", ["games/gamestate.py"]),
    Extension("games.amazons", ["games/amazons.py"], include_dirs=[numpy.get_include()]),
    Extension("games.tictactoe", ["games/tictactoe.py"], include_dirs=[numpy.get_include()]),
    Extension("games.breakthrough", ["games/breakthrough.py"], include_dirs=[numpy.get_include()]),
    Extension("games.kalah", ["games/kalah.py"]),
    Extension("games.blokus", ["games/blokus.py"], include_dirs=[numpy.get_include()]),
    Extension("ai.transpos_table", ["ai/transpos_table.py"], include_dirs=[numpy.get_include()]),
    Extension("ai.alpha_beta", ["ai/alpha_beta.pyx"]),
    Extension("ai.mcts", ["ai/mcts.py"]),
]

setup(name="Cython MCTS", ext_modules=cythonize(ext_modules))
