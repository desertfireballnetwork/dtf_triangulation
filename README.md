# dtf_triangulation
The Dynamic Trajectory Fit (DTF) method is a unique meteoroid triangulation technique that utilises the meteoroid's dynamic equations of motion to determine both dynamic and physical parameters of a meteoroid directly from the observed line-of-sight information. More details on the inner workings of the DTF method can be found at https://arxiv.org/abs/1911.00816 (A Dynamic Trajectory Fit to Multi-Sensor Fireball Observations).

## Installation
Before you can run the DTF software on your PC, there are some required modules that must be externally installed, including the following:
> pip3 install numpy, scipy, astropy, matplotlib, pyyaml

> sudo apt install python3-tk

Additionally, you have to download the python files from this repo and place it into a local folder on your PC.

## Run
To run the DTF algorithm locally, you can run the following code in the terminal:
> cd /path/to/saved/python/files/

> python3 DynamicTrajectoryFit.py -d data/

This should produce the fitted trajectory data (ecsv format), trajectory plots (including along-track error, cross-track error, modelled beta, modelled mass, modelled velocity), and a variety of Google Earth files to visualise the trajectory (kml format).

## Outputs
A variety of triangulation outputs have been provided in the "outputs" folder, including the Method of Planes (MOP: Ceplecha, 1987), the Straight Line Least Squares method (SLLS: Borovicka, 1990), the Multi-Parameter Fit (MPF: Gural, 2012), the Dynamic Trajectory Fit (DTF: Jansen-Sturgeon, 2019), and Dynamic Trajectory Fit with Fragmentation (DTFrag: Jansen-Sturgeon, 2019). A few plots comparing these methods can be found in the "outputs/triangulation_method_comparisons" folder. 
