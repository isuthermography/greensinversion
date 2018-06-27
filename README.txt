Greensinversion is a package for model-based inversion of
flash thermography measurement data.

PREREQUISITES
-------------
 * Fast computer with moderately high-end GPU (suggest at least
   1GB video ram), multiple cores,  and lots of memory (128GB minimum)
 * greensconvolution package and all its prerequisites, including:
 
Python -- Tested with Python 2.7; should work with Python 3.x
          but might need minor compatibility bugfixes
Numpy  -- Any recent version should be fine
Scipy  -- Any recent version should be fine
Cython -- Any recent version should be fine. Cython will need to be
          configured with a suitable C compiler. On Linux this is
	  usually handled by your package manager. On Windows, see
	  https://github.com/cython/cython/wiki/installingonwindows and
	  https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
OpenCL -- You will also to have the OpenCL installable client driver
	  available. On Linux this is usually as simple as 
	  "dnf install ocd-icd-devel". On Windows make sure you have
	  the OpenCL drivers provided by your GPU card vendor installed
PyOpenCL -- From https://mathema.tician.de/software/pyopencl/
          On Linux this may be available with your package manager,
	  e.g. "dnf install python2-pyopencl".
NetCDF4 Python bindings -- http://unidata.github.io/netcdf4-python/. On
                           Linux this may be as simple as
			   "dnf install python2-netcdf4"

INSTALLATION
------------
To build greensinversion:
  python setup.py build

To install into site-packages (may need to be root or Administrator)
   python setup.py install

TESTING/VERIFICATION
--------------------
Run the demos/verification.py script and make sure
that none of the assertions fail.

RUNNING MODEL-BASED INVERSION
-----------------------------
Most people will want to start from the commented example:
demos/greensinversion_inverse_demo_TWIRAW.py

  * Be sure to program in suitable values for density (rho)
    and specific heat (c) for your material, as well as
    suitable values for the through-thickenss and in-plane
    diffusivities (alphaz and alphaxy, respectively)
  * Be sure to evaluate the spatial pixel size of your thermal
    images and assign dx and dy (in meters)
  * Be sure to select spatial downsampling if your thermal
    resolution is very high (much higher than 1 pixel/mm).
    Otherwise the compute workload might be extreme.
  * Be sure to select which OpenCL device to use. You can
    use the "clinfo" command to list available devices.
    If you don't specify the example will pick the first GPU
    it sees.
  * If analyzing data from a composite,
    adjust nominal_lamina_thickness
  * Set the layer z positions and reflector densities
    (reflectors variable)
  * May need to increase numplotrows and numplotcols if
    the number of reflecting layers is increased.
  * Adjust frames_to_discard from looking at your raw
    data and finding how many frames (including the frame
    marked as the flash) are contaminated by saturation.
    These will be ignored in the inversion process.
  * Include appropriate code to load in your thermal
    image sequence, and determine the time of the first
    frame, the timestep, the frame index (0-based) of the
    frame with the flash, and a data array, with time,
    y, and x indices. The example reads uncalibrated
    data from a Thermal Wave Imaging .RAW file.
    It is best if your data is calibrated in degrees Kelvin.
    That way the power fields plotted will be in units of J/m^2.
    Otherwise your output power will not be in meaningful units.
  * Mask out regions in the thermal image data beyond
    the boundaries of the sample as NaN. This will
    reduce edge effects and accelerate the computation. 
  * You will need to run it once to determine a suitable
    Tikhonov parameter for your setup. The Tikhonov parameter
    is dependent on your camera noise level and on the
    scaling of the measured temperatures. 
    If you have a cooled InSb camera and calibrated temperature
    measurements (in deg. K) then our value of 7.5e-11 would
    be a good starting point. Otherwise the first plot generated
    when running the example is a diagnostic plot for the
    regularization process. You will want to select the Tikhonov
    parameter to match the y value on that plot above where it
    transitions from vertical to horizontal.
  * Set the plot_min_power_per_area and plot_max_power_per_area
    to see a reasonable color scaling of the plots. For reasonable
    flash lamps and calibrated temperatures in deg. K, we have
    found that -10000 to +30000 J/m^2 is a good range.


ACKNOWLEDGMENTS
---------------
If using or building on this software please be sure to cite the authors!
S.D. Holland and B. Schiefelbein, Model-based inversion for pulse thermography, J. Exp. Mech (under review, 2018)

This material is based own work supported by NASA through Early
Stage Innovation grant #NNX15AD75G.



Copyright (C) 2015-2018 Iowa State University Research Foundation, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

