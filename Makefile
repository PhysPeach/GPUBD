test: testsrc/test.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

%.o: %.cu h/%.cuh h/parameters.cuh
	nvcc -arch=sm_60 -o $@ -c $<

%.o: %.cu h/parameters.cuh
	nvcc -arch=sm_60 -o $@ -c $<