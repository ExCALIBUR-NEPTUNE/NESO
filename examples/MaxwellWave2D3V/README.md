# Electromagnetic 2D3V magnetoacoustic cyclotron instability example

These are the input files to run and plot the two stream instability example.

To run the executable first install `NESO` which will build the `MaxwellWave2D3V` solver. After `NESO` is built the solver can be ran as follows:

```
OMP_NUM_THREADS=1 mpirun -n 12 MaxwellWave2D3V2D3V conditions.xml mesh.xml
```

This should produce an output `MaxwellWave2D3V2D3V_field_trajectory.h5`.

