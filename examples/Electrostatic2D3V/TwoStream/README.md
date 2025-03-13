# Electrostatic 2D3V Two Stream Instability Example

These are the input files to run and plot the two stream instability example. The underlying theory is described in report `M4c: 1-D and 2-D particle models`.

To run the executable first install `NESO` which will build the `Electrostatic2D3V` solver. After `NESO` is built the electrostatic solver can be ran as follows:

```
OMP_NUM_THREADS=1 mpirun -n 12 Electrostatic2D3V two_stream_conditions.xml two_stream_mesh.xml
```

This should produce an output `Electrostatic2D3V_field_trajectory.h5` which can be plotted with

```
pip install -r requirements.txt
python3 plot_two_stream_energy.py two_stream_conditions.xml electrostatic_two_stream.h5
```
