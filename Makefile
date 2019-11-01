test: testsrc/test.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

measuretime: measuresrc/timer.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD1024: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD2048: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD4096: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD16384: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD32768: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD65536: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD131k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD262k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD524k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

BD1048k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

%.o: %.cu h/%.cuh h/parameters.cuh
	nvcc -arch=sm_60 -o $@ -c $<

%.o: %.cu h/parameters.cuh
	nvcc -arch=sm_60 -o $@ -c $<