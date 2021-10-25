# LIRA  (SIMD version)

Low-count Image Reconstruction and Analysis 



## What's new
[LIRA](https://github.com/astrostat/LIRA/)'s C code is re-written in C++ with SIMD (Single Instruction, Multiple Data)intrinsics from Google's [highway](https://github.com/google/highway/) library for improved speed. Depending on the processor, the speed gains can be at least 3x-4x. In addition, the following new options can be set:
* Re-sample the PSF from the original input once every `n` iterations
* Switch the working precision between float/double. Note: For larger input sizes, setting a lower precision can increase the speed even further.
* Planned updates
  * Create R and Python packages.
  * Introduce OpenMP directives for computationally expensive loops.


## Build instructions
### R
--TBU--

### Python
--TBU--


## References

Stein, N.M., van Dyk, D.A., Kashyap, V.L., & Siemiginowska, A., Detecting Unspecified Structure in Low-Count Images, 2015, ApJ, 813, 66

Connors, A., Stein, N.M., van Dyk, D., Kashyap, V., & Siemiginowska, A., 2011, ADASS XX, ASPC (Eds. Ian N. Evans, Alberto Accomazzi, Douglas J. Mink, and Arnold H. Rots), v442, p463 (2011ASPC..442..463C)

Connors, A. & van Dyk, D.A., 2007, SCMA IV, ASPC (Eds. G.J.Babu and E.D.Feigelson), v371, p101 (2007ASPC..371..101C)

Esch, D.N., Connors, A., Karovska, M., & van Dyk, D.A., 2004, ApJ, 610, 1213