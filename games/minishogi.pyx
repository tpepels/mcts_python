# minishogi_ext.pyx
from cython.cimports.libc.stdint import int32_t
from minishogi cimport MiniShogi  # Import the class from the .py file

# TODO Hier was je gebleven, het probleem is dat de callbacks enorme python objecten zijn die je telkens aanmaakt
# TODO Je kan geen functiepointers gebruiken in pure python, dus je moet een andere manier vinden om dit te doen