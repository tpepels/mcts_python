from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


import numpy

# compile_args = ["-O3", "-DCYTHON_WITHOUT_ASSERTIONS"]
# * Switch comments to disable/enable optimizations
compile_args = []

ext_modules = [
    Extension(
        "includes.gamestate",
        ["includes/gamestate.pyx"],
        extra_compile_args=compile_args,
    ),
    Extension(
        "includes.c_util",
        ["includes/c_util.pyx"],
        extra_compile_args=compile_args,
    ),
    Extension(
        "games.amazons",
        ["games/amazons.py"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "games.tictactoe",
        ["games/tictactoe.py"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "games.breakthrough",
        ["games/breakthrough.py"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "games.minishogi",
        ["games/minishogi.py"],
        extra_compile_args=compile_args,
    ),
    Extension(
        "ai.transpos_table",
        ["ai/transpos_table.py"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "ai.alpha_beta",
        ["ai/alpha_beta.pyx"],
        include_dirs=["."],
        extra_compile_args=compile_args,
    ),
    Extension(
        "ai.mcts",
        ["ai/mcts.py"],
        include_dirs=["."],
        extra_compile_args=compile_args,
    ),
    Extension(
        "run_games",
        ["run_games.py"],
        extra_compile_args=compile_args,
    ),
    Extension(
        "includes.dynamic_bin",
        ["includes/dynamic_bin.py"],
        extra_compile_args=compile_args,
    ),
    Extension(
        "playout_test",
        ["playout_test.py"],
        extra_compile_args=compile_args,
    ),
    # Extension("games.blokus", ["games/blokus.py"], include_dirs=[numpy.get_include()]),
    # Extension("games.kalah", ["games/kalah.py"]),
]

setup(
    name="Cython MCTS",
    ext_modules=cythonize(
        ext_modules,
        annotate=True,
        compiler_directives={
            # "profile": True,
            "overflowcheck": False,
            "language_level": "3",
            "embedsignature": True,
            "wraparound": False,
            "initializedcheck": False,
            "boundscheck": False,
            "nonecheck": False,
            "cdivision": True,
            "infer_types": True,
            "cdivision_warnings": False,
        },
    ),
)
