import sys
import os 
import os.path
import collections

import numpy as np


def RAWfile_ReadHeaderLine(fh):
    # Header is text, each line terminated with \x0d\x0a
    # Return line with line terminator stripped, or None
    # to indicate header is over
    
    byte=fh.read(1)
    accum=b""
    while ord(byte) != 0 and byte!=b'\x0a':
        accum+=byte
        byte=fh.read(1)
        pass

    if ord(byte)==0:
        return None
    
    assert(byte==b'\x0a') # linefeed: Only other way to escape above loop
    assert(accum[-1:]==b'\x0d') # previous character should be a CR
    return accum[:-1]

def ReadRAWHeader(fh):
    formatline=RAWfile_ReadHeaderLine(fh)

    assert(formatline==b"EchoTherm Raw Format 4.0") # Only tested on v4.0 (?)
    
    # Read 'Parameter=Value' pairs
    Params=collections.OrderedDict()
    
    HeaderLine=RAWfile_ReadHeaderLine(fh)
    
    while HeaderLine is not None:
        assert(b'=' in HeaderLine)

        (ParamName,ParamValue)=HeaderLine.split(b'=',1)
        Params[ParamName.decode('utf-8')]=ParamValue.decode('utf-8')
        
        HeaderLine=RAWfile_ReadHeaderLine(fh)
        pass
    
    return Params

def ReadRAW(filepath):
    filesize=os.path.getsize(filepath)

    fh=open(filepath,"rb")

    HeaderParams=ReadRAWHeader(fh)
    

    # Process parameters
    assert(HeaderParams["Bits/Pixel"]=="16")
    #numpy_dtype=np.uint16
    numpy_dtype='H'
 
    dt=1.0/float(HeaderParams["Capture Frequency"])

    flashframe=int(HeaderParams["FlashFrame"])-1 # NOTE: FlashFrame is numbered from 1, not zero; our flashframe variable is numbered from 0
    
    t0=-dt*flashframe  


    

    dimlen=np.array([int(HeaderParams["Frames"]),int(HeaderParams["Height"]),int(HeaderParams["Width"])],dtype=np.uint64)

    expectedsize=int(np.prod(dimlen))*2
    offset=filesize-expectedsize # We ought to be able to get this from the 'Data Offset' parameter, but it seems to be too low by 900 (?)
    
    fh.seek(offset)
    
    # Reverse y axis so dataguzzler image looks right
    data=np.empty(dimlen,dtype=np.float32)
    data[::]=np.fromfile(fh,dtype=numpy_dtype).reshape(tuple(dimlen),order='C')

    # Data is indexed C-style raster scan, indexed frame #, row # , column #
    return (t0,dt,flashframe,HeaderParams,data)

