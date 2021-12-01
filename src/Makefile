all:
	dpcpp mesh.cpp plasma.o -c
	dpcpp plasma.cpp mesh.o -c
	dpcpp main.cpp plasma.o mesh.o
