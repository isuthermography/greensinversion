import sys
import os
import os.path
import subprocess
import re
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import setuptools.command.bdist_egg
import sys
import distutils.spawn
import numpy as np
from Cython.Build import cythonize

#from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension

extra_compile_args = {
    "msvc": ["/openmp","/Dfloat64_t=double"],
    #"unix": ["-O0", "-g", "-Wno-uninitialized"),    # Replace the line below with this line to enable debugging of the compiled extension
    "unix": ["-fopenmp","-O5","-Wno-uninitialized"],
    "clang": ["-fopenmp","-O5","-Wno-uninitialized"],
}

extra_include_dirs = {
    "msvc": [".", np.get_include() ],
    "unix": [".", np.get_include() ],
    "clang": [".", np.get_include() ],
}

extra_libraries = {
    "msvc": [],
    "unix": ["gomp",],
    "clang": [],
}

extra_link_args = {
    "msvc": [],
    "unix": [],
    "clang": ["-fopenmp=libomp"],
}


class build_ext_compile_args(build_ext):
    def build_extensions(self):
        compiler=self.compiler.compiler_type
        for ext in self.extensions:
            if compiler in extra_compile_args:
                ext.extra_compile_args=extra_compile_args[compiler]
                ext.extra_link_args=extra_link_args[compiler]
                ext.include_dirs.extend(list(extra_include_dirs[compiler]))
                ext.libraries.extend(list(extra_libraries[compiler]))
                pass
            else:
                # use unix parameters as default
                ext.extra_compile_args=extra_compile_args["unix"]
                ext.extra_link_args=extra_link_args["unix"]
                ext.include_dirs.extend(list(extra_include_dirs["unix"]))
                ext.libraries.extend(extra_libraries["unix"])
                pass
                
            pass
            
        
        build_ext.build_extensions(self)
        pass
    pass


greensinversion_package_files = [ "pt_steps/*" ]

ext_modules=cythonize("greensinversion/*.pyx")

emdict=dict([ (module.name,module) for module in ext_modules])

#gi_fillholes_pyx_ext=emdict['greensinversion.fillholes']
#gi_fillholes_pyx_ext.sources.append("greensinversion/fillholes_c.c")
#gi_fillholes_pyx_ext.extra_compile_args=['-g']
#gi_fillholes_pyx_ext.extra_compile_args=['-fopenmp','-O3']
#gi_fillholes_pyx_ext.extra_link_args=['-lgomp']



setup(name="greensinversion",
      description="greensinversion",
      author="Stephen D. Holland",
      url="http://thermal.cnde.iastate.edu",
      zip_safe=False,
      ext_modules=ext_modules,
      packages=["greensinversion"],
      cmdclass={
          "build_ext": build_ext_compile_args,
      },
      package_data={"greensinversion": greensinversion_package_files},
      entry_points={"limatix.processtrak.step_url_search_path": [ "limatix.share.pt_steps = greensinversion:getstepurlpath" ]})
