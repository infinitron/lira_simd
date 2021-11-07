LIRA  (SIMD version)
====================

Low-count Image Reconstruction and Analysis 



## What's new
[LIRA](https://github.com/astrostat/LIRA/)'s C code is re-written in C++ with SIMD (Single Instruction, Multiple Data) intrinsics using Google's [highway](https://github.com/google/highway/) library for improved speed. Depending on the processor, the speed gains can range from 3x-4x. Additionally, the working precision can be switched from double to float, which can improve the speed even further on larger images.
* Planned updates
  * R and ~~python~~ packages.
  * Introduce OpenMP directives for computationally expensive loops.
  * Ability to introduce a new PSF once every `n` iterations.
  * Multi-PSF (e.g., energy-based) reconstruction,

### [New] Python package
[pylira_simd](https://github.com/infinitron/pylira_simd)

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