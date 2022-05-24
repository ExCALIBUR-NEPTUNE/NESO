# Dockerfile for running NESO

This dockerfile provides Intel's OneAPI, plus basic tools, intended for running NESO.

To build the image, do

```
docker build -t NESOenv .
```

and to execute, do

```
docker run --rm -it NESOenv
```

You may find it useful to use a colour terminal, and to mount the directories containing your ssh keys and software,
in which case, execute with

```
docker run --rm -it -e "TERM=xterm-256color" -v ~/.ssh/:/root/.ssh -v ~/<code_dir>:/root/<code_dir> NESOenv
```
