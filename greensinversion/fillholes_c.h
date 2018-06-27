void fillholes_flat_c(float *outputmat,uint8_t *maskarray,size_t nt,size_t ny,size_t nx)
{
  size_t tcnt;
#ifdef _MSC_VER
  uint64_t infinity_val=0x000000007FF00000l;
  uint64_t neginfinity_val=0x00000000FFF00000l;
  double infinity_dbl,neginfinity_dbl;

  neginfinity_dbl = *(float64_t *)&neginfinity_val; /* -infinity */
  infinity_dbl = *(float64_t *)&infinity_val; /* infinity */
#else 
  double neginfinity_dbl=-1.0/0.0; /* -infinity */
  double infinity_dbl=1.0/0.0; /* infinity */
#endif

#ifdef USE_OPENMP
#pragma omp parallel for shared(outputmat,maskarray,nt,ny,nx,infinity_dbl,neginfinity_dbl) default(none) private(tcnt)
#endif /* USE_OPENMP */
  for (tcnt=0;tcnt < nt; tcnt++) {
    /* This works by solving Laplace's equation del^2 T = 0 
       for all of the unmasked elements to get reasonable values. 
       it is performed independently at each time, which could
       give some weird side effects */

    /* Solve laplace's equation by averaging adjacent neighbors 
     repeatedly until convergence */

    /* see http://www.public.iastate.edu/~akmitra/aero361/design_web/Laplace.pdf */
    /* We are a little loose, not keeping a store of our "new version" separate from the "old version", but the converged answer has to be the same, so 
       we don't care. */

    /* could be sped up by implementing "relaxation" (see web link, above). 
       To be legitimate in that case would probably want to separate
       new and old versions properly */
    
    size_t xcnt,ycnt;
    size_t numavg;
    float outputval,inputval;
    float maxout;
    float minout;
    float maxchange;
    float diff;
	
    do  {

      maxout=neginfinity_dbl;
      minout=infinity_dbl;

      maxchange=neginfinity_dbl;
      
      for (xcnt=0;xcnt < nx;xcnt++) {
	for (ycnt=0;ycnt < ny;ycnt++) {
	  if (!maskarray[nx*ycnt + xcnt]) {
	    outputval=0.0;
	    numavg=0;

	    if (ycnt < ny-1) {
	      inputval=outputmat[ny*nx*tcnt + nx*(ycnt+1) + xcnt];
	      if (isfinite(inputval)) {
		// neighbor at (xcnt,ycnt+1) is valid
		outputval += inputval;
		numavg++;
	      }
	    }
	    
	    if (xcnt < nx-1) {
	      inputval=outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt+1];
	      if (isfinite(inputval)) {
		// neighbor at (xcnt+1,ycnt) is valid
		outputval += inputval;
		numavg++;
	      }
	    }

	    
	    if (ycnt > 0) {
	      inputval=outputmat[ny*nx*tcnt + nx*(ycnt-1) + xcnt];
	      if (isfinite(inputval)) {
		// neighbor at (xcnt,ycnt-1) is valid
		outputval += inputval;
		numavg++;
	      }
	    }

	    if (xcnt > 0) {
	      inputval=outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt-1];
	      if (isfinite(inputval)) {
		// neighbor at (xcnt-1,ycnt) is valid
		outputval += inputval;
		numavg++;
	      }
	    }

	    if (numavg > 0) {
	      outputval /= numavg;

	      if (isfinite(outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt])) {
		diff = fabsf(outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt]-outputval);
	      } else {
		diff = infinity_dbl;
	      }
	      
	      outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt]=outputval;

	      if (diff > maxchange) {
		maxchange=diff;
	      }
	    }
	    
	  }
	  if (isfinite(outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt]) && maxout < outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt]) {
	    maxout=outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt];
	  }
	  if (isfinite(outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt]) && minout > outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt]) {
	    minout=outputmat[ny*nx*tcnt + nx*(ycnt) + xcnt];
	  }
	  
	  
	}
	
      }
	
      /* iterate while we are making any change and the 
	 change is > 1e-3 of the peak difference */
    } while (maxchange >= 0 && maxchange > 1e-3 * (maxout-minout));
    
  }
  
}
