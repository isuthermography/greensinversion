import sys
from Cython.Build import cythonize
from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension

ext_modules=cythonize("greensinversion/*.pyx")

emdict=dict([ (module.name,module) for module in ext_modules])

gi_fillholes_pyx_ext=emdict['greensinversion.fillholes']
#gi_fillholes_pyx_ext.sources.append("greensinversion/fillholes_c.c")
#gi_fillholes_pyx_ext.extra_compile_args=['-g']
gi_fillholes_pyx_ext.extra_compile_args=['-fopenmp','-O3']
gi_fillholes_pyx_ext.extra_link_args=['-lgomp']


numpy_setup(name="greensinversion",
            description="greensinversion",
            author="Stephen D. Holland",
            url="http://thermal.cnde.iastate.edu",
            ext_modules=ext_modules,
            packages=["greensinversion"])
