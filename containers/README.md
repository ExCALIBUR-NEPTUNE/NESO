# Contiainers for running NESO

## Singularity

First build the base image in the `containers` directory

```
singularity build sycl-base.sif sycl-base.def
```

then build the images as per required sycl implementation.

**hipsycl**

```
singularity build hipsycl.sif hipsycl.def
```

**oneapi-sycl**

```
singularity build oneapi-sycl.sif oneapi-sycl.def
```

Then run with

```
sudo singularity run --hostname neso-singularity --bind ~/.ssh/:/root/.ssh --bind ~/<code_dir>:/root/<code_dir> <image>.sif
```

Within the containers `module avail` will display e.g. `neso-hipsycl` or `neso-oneapi` which can be loaded to build NESO.

## Docker

### OneAPI sycl

This dockerfile provides Intel's OneAPI, plus basic tools, intended for running NESO.

To build the image, do

```
docker build -t nesoenv .
```

and to execute, do

```
docker run --rm -it nesoenv
```

You may find it useful to use a colour terminal, and to mount the directories containing your ssh keys and software,
in which case, execute with

```
docker run --rm -it -e "TERM=xterm-256color" -v ~/.ssh/:/root/.ssh -v ~/<code_dir>:/root/<code_dir> nesoenv
```
