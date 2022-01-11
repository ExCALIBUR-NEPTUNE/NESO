#include "mesh.hpp"
#include "plasma.hpp"
#include "diagnostics.hpp"

int main();
void initialize();
void evolve(Mesh *mesh, Plasma *plasma, FFT *fft, Diagnostics *diagnostics);
