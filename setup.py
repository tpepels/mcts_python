from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext


import numpy

ext_modules = [
    Extension("games.amazons", ["games/amazons.py"], include_dirs=[numpy.get_include()]),
    Extension("games.tictactoe", ["games/tictactoe.py"], include_dirs=[numpy.get_include()]),
    Extension("games.breakthrough", ["games/breakthrough.py"], include_dirs=[numpy.get_include()]),
    Extension("games.kalah", ["games/kalah.py"]),
    Extension("games.blokus", ["games/blokus.py"], include_dirs=[numpy.get_include()]),
    Extension("ai.transpos_table", ["ai/transpos_table.py"], include_dirs=[numpy.get_include()]),
    Extension("ai.alpha_beta", ["ai/alpha_beta.pyx"]),
    Extension("ai.mcts", ["ai/mcts.py"]),
    Extension("c_util", ["c_util.pyx"]),
    # Extension("games.gamestate", ["games/gamestate.pyx"]),
]

setup(
    name="Cython MCTS",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "profile": True,
            "language_level": "3",
            "embedsignature": True,
            "initializedcheck": False,
            "boundscheck": False,
            "nonecheck": False,
            "cdivision": True,
            "infer_types": True,
        },
        build_dir="build",
    ),
    packages=["ai", "games"],
    package_data={"games": ["games/gamestate.pxd"]},
)
