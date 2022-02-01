all:
	dpcpp fft.cpp -c -g
	dpcpp mesh.cpp -c -g
	dpcpp plasma.cpp -c -g
	dpcpp diagnostics.cpp -c -g
	dpcpp simulation.cpp -c -g
	dpcpp main.cpp plasma.o mesh.o fft.o diagnostics.o simulation.o -L/usr/lib/x86_64-linux-gnu/ -llapack -g -lfftw3 -lm
