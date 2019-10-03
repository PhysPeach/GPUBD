test: testsrc/test.o c/particles.o c/box.o c/grid.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

%.o: %.cu h
	nvcc -arch=sm_60 -o $@ -c $<

MT.o: MT.cu h/MT.h
	nvcc -arch=sm_60 -o $@ -c $<