# Building the documentations

## Website

The website is built into the output directory `docs/build`. 
The website is build by:
1. Build the Doxygen output following the Doxygen instructions below.
2. Build the Sphinx output following the Sphinx instructions below.
3. Copy the Sphinx html source from `docs/build/sphinx/html` to your website root.
4. Copy the Doxygen html source from `docs/build/doxygen` to your website root.

For reference see the github actions script at `.github/workflows/build_docs.yaml`.

## Doxygen

1. Install Doxygen, Graphviz
2. Run `make` in `docs/doxygen` output will be in `docs/build`.

## Sphinx

1. Install a working python3/pip environment.
2. Run `pip3 install -r requirements.txt` in `docs/sphinx`.
3. Run `make html` in `docs/sphinx`.

## PDFs

PDF documents can be generated from the markdown files in the `docs/` directory via `pandoc`.

```
pandoc docs/Foo.md -o Foo.pdf
```
