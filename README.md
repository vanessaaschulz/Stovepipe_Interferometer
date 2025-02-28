# Stovepipe Interferometer

This Repository contains Python scripts and Juypter Notebooks to teach concepts of radio interferometry and analyse data from a specific setup (Stovepipe Interferometer).

## Description
The folder 3\_5\_Preparation\_Tests\_for\_Analysis\_Procedure includes four sub folders dealing with different concepts of 
radio interferometry using Jupyter Notebooks:
* Modeling uv Coordinates for a 3 element interferometer
* Modeling the fringe period for a 3 element interferometer
* Modeling the Point Spread Function (PSF) for a 3 element Interferometer
* Testing the CLEAN algorithm to deconvolve the dirty map

The folder 3\_6\_Cantenna\_Interferometer\_Analysis\_in\_Steps contains several python scripts and one Jupyter Notebook which 
calls upon the python scripts. This is used to analyse data collected with a 3 element transit interferometer:
* power signals of 3 antennas (RMS and HPBW analysis)
* Complex correlation
* Calculating Complex Visibility (Amplitude and Phase)
* Calculating the UV Coverage
* Deriving a Dirty Map, snapshot-PSF and Dirty Beam
* Aplly CLEAN algorithm
* Estimate fringe period and compare to modeled fringe period

