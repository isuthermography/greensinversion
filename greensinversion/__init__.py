import sys
import os
import os.path

try:
    # py2.x
    from urllib import pathname2url
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    pass


class dummy(object):
    pass

pkgpath = sys.modules[dummy.__module__].__file__
pkgdir=os.path.split(pkgpath)[0]

def getstepurlpath():

    return [ pathname2url(os.path.join(pkgdir,"pt_steps")) ]

from .sourcevecs import scaledcondition as scaledcondition
from .sourcevecs import build_flash_source_vecs as build_flash_source_vecs
from .sourcevecs import build_reflector_source_vecs as build_reflector_source_vecs
from .sourcevecs import build_reflector_source_vecs_curved as build_reflector_source_vecs_curved
from .sourcevecs import build_all_source_vecs as build_all_source_vecs
from .inversion import timelimitmatrix as timelimitmatrix
from .inversion import generateinversionsteps as generateinversionsteps
from .inversion import performinversionsteps as performinversionsteps
from .inversion import generatesinglestepinversion as generatesinglestepinversion
from .inversion import plotabstractinverse as plotabstractinverse
from .inversion import buildconcreteinverse as buildconcreteinverse
from .inversion import plotconcreteinverse as plotconcreteinverse
from .inversion import define_curved_inversion as define_curved_inversion
from .inversion import define_flat_inversion as define_flat_inversion
from .inversion import savetiledconcreteinverse as savetiledconcreteinverse
from .inversion import saturationcheck as saturationcheck
from .inversion import setupinversionprob as setupinversionprob
from .inversion import num_sources as num_sources
from .inversion import perform_flat_inversion as perform_flat_inversion

from .tile_rectangle import build_tiled_rectangle as build_tiled_rectangle


from .grid import build_gi_grid,build_gi_grid_3d    

from .inversion import NotANumberError


