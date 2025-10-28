This serves to aim as a guide for next steps

1. (Completed) remove big if/elif from optimize.py

2. Function to extract at any given stage to calculate scatter
   
2.  Function to find minimum of residual at a given cost aka. dppm/dstep=0

3. Attach DQ flags (in any raw fits file there should be associated hdu's like [2,3] that have DQ flags and Grp flags,
   add before 1/f step and if there is a step before that,
 check with MR if DQ flags are needed pre-jump.

5. Implement above changes on first integration in list as the optimizing data. 
