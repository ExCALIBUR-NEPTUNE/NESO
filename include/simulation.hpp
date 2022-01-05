#include <iostream>

#include "mesh.hpp"
#include "plasma.hpp"

void initialize();
void evolve(Mesh *mesh, Plasma *plasma, FFT *fft);
