from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


import numpy

ext_modules = [
    Extension("includes.gamestate", ["includes/gamestate.pyx"]),
    Extension("includes.c_util", ["includes/c_util.pyx"]),
    Extension("games.amazons", ["games/amazons.py"], include_dirs=[numpy.get_include()]),
    Extension("games.tictactoe", ["games/tictactoe.py"], include_dirs=[numpy.get_include()]),
    Extension("games.breakthrough", ["games/breakthrough.py"], include_dirs=[numpy.get_include()]),
    Extension("games.kalah", ["games/kalah.py"]),
    Extension("games.blokus", ["games/blokus.py"], include_dirs=[numpy.get_include()]),
    Extension("ai.transpos_table", ["ai/transpos_table.py"], include_dirs=[numpy.get_include()]),
    Extension("ai.alpha_beta", ["ai/alpha_beta.pyx"], include_dirs=["."]),
    Extension("ai.mcts", ["ai/mcts.py"], include_dirs=["."]),
    Extension("run_games", ["run_games.py"]),
]

setup(
    name="Cython MCTS",
    ext_modules=cythonize(
        ext_modules,
        annotate=True,
        compiler_directives={
            "profile": True,
            "language_level": "3",
            # "embedsignature": True,
            # "wraparound": False,
            # "initializedcheck": False,
            # "boundscheck": False,
            # "nonecheck": False,
            # "cdivision": True,
            # "infer_types": True,
        },
    ),
)
