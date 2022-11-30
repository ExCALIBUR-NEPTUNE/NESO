# Building the documentations

## Website

The website is built into the output directory `docs/build`. 
The website is built by:
1. Installing the dependencies listed below for Doxygen and Sphinx
2. Running `make` in the `docs` directory, this will build the documentation for the current branch in `docs/build`.

For reference see the github actions script at `.github/workflows/build_docs.yaml`.

## Doxygen

1. Install Doxygen, Graphviz

## Sphinx

1. Install a working python3/pip environment.
2. Run `pip3 install -r requirements.txt` in `docs/sphinx`.

## PDFs

PDF documents can be generated from the markdown files in the `docs/` directory via `pandoc`.

```
pandoc docs/Foo.md -o Foo.pdf
```
