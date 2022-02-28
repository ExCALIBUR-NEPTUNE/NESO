all:
	dpcpp fft.cpp -c -g
	dpcpp mesh.cpp -c -g -fsycl
	dpcpp diagnostics.cpp -c -g -fsycl
	dpcpp simulation.cpp -c -g
	dpcpp species.cpp -c -g -fsycl
	dpcpp plasma.cpp -c -g -fsycl
	dpcpp main.cpp plasma.o mesh.o fft.o diagnostics.o simulation.o species.o -L/usr/lib/x86_64-linux-gnu/ -llapack -g -lfftw3 -lm
