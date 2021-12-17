all:
	dpcpp mesh.cpp -c -g
	dpcpp plasma.cpp -c -g
	dpcpp main.cpp plasma.o mesh.o -L/usr/lib/x86_64-linux-gnu/ -llapack -g
