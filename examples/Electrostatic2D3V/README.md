# The Electrostatic2D3V solver

## Overview (unfinished)

The theory behind the Electrostatic2D3V solver and the `two_stream` example can be found in NEPTUNE report [M4c.1: 1-D and 2-D Particle Models](https://excalibur-neptune.github.io/Documents/CD-EXCALIBUR-FMS0070-1.00-M4c.1_ExcaliburFusionModellingSystem.html)

## Examples

### two_stream (unfinished)

To run the executable first install `NESO` which will build the `Electrostatic2D3V` solver. After `NESO` is built the electrostatic solver can be ran as follows:

```
OMP_NUM_THREADS=1 mpirun -n 12 Electrostatic2D3V two_stream.xml rectangle1x0.02_20x1quads.xml
```

This should produce an output `Electrostatic2D3V_field_trajectory.h5` which can be plotted with

```
pip install -r requirements.txt
python3 plot_two_stream_energy.py two_stream.xml electrostatic_two_stream.h5
```
