test: testsrc/test.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

measuretime: measuresrc/timer.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD1024: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD2048: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD4096: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD8192: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD16384: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD32768: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD65536: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD131k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD262k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD524k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

MD1048k: src/main.o c/particles.o c/box.o c/grid.o c/generalFuncs.o c/MT.o
	nvcc -arch=sm_60 -o $@ $^

%.o: %.cu h/%.cuh h/parameters.cuh
	nvcc -arch=sm_60 -o $@ -c $<

%.o: %.cu h/parameters.cuh
	nvcc -arch=sm_60 -o $@ -c $<