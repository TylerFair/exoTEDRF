This serves to aim as a guide for next steps

1. (Completed) remove big if/elif from optimize.py

2. Function to box extract at any given stage to calculate scatter. Use default box aperture widths for each instrument. 
   
2.  Function to find minimum of residual at a given cost aka. dppm/dstep=0, would prefer gradient descent over np.min(). 

3. Attach DQ flags (in any raw fits file there should be associated hdu's like [2,3] that have DQ flags and Grp flags to each step. Eg. np.nansum(column)/np.sum(np.isfinite(column)) after making flags NaN. 

5. Implement above changes on first segment in list as the optimizing data. 
