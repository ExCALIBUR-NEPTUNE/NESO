all:
	dpcpp fft.cpp -c -g
	dpcpp mesh.cpp -c -g -fsycl
	dpcpp diagnostics.cpp -c -g -fsycl
	dpcpp simulation.cpp -c -g
	dpcpp species.cpp -c -g -fsycl
	dpcpp plasma.cpp -c -g -fsycl
	dpcpp main.cpp plasma.o mesh.o fft.o diagnostics.o simulation.o species.o -L/usr/lib/x86_64-linux-gnu/ -llapack -g -lfftw3 -lm -L/opt/intel/oneapi/mkl/2022.0.2/lib/intel64/ -I/opt/intel/oneapi/mkl/2022.0.2/include/  -DMKL_ILP64  -I"${MKLROOT}/include"  -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
