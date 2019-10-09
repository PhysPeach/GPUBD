#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "../h/MT.h"
#include "../h/particles.cuh"
#include "../h/grid.cuh"
#include "../h/box.cuh"
#include "../h/parameters.cuh"

unsigned int IT;
unsigned int IDs;
unsigned int IDe;
float tau;
float Tfin;

using namespace PhysPeach;

__global__ void setVec(float* x){
    uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint i = i_global; i < D * NP; i += NB * NT){
        x[i] = 1;
        if(i == NP){
            x[i] = 50.;
        }
    }
}

int main(){
    std::cout << "hello, test" << std::endl;

    //test
    Tfin = 1;
    tau = 100;
    IDs = 1;
    IDe = 1;

    //initialise random func
    init_genrand((unsigned long)time(NULL));

    std::cout << "---Settings---" << std::endl;
    std::cout << "Tfin = " << Tfin << std::endl;
    std::cout << "t_eq = " << tau << std::endl;
    std::cout << "t_rec = " << tau << std::endl;
    std::cout << "ID = [" << IDs << ", " << IDe << "]" << std::endl;
    std::cout << "--------------" << std::endl;

    float *vtest_dev;
    cudaMalloc((void**)&vtest_dev, D * NP * sizeof(float));
    //x[NP] = 50, else 1
    setVec<<<NB,NT>>>(vtest_dev);

    Grid g;
    makeGrid(&g, 36.);
    setUpdateFreq(&g, 0.001, vtest_dev);
    std::cout << g.updateFreq << " = " << a0/(50. * 0.001) << std::endl;
    killGrid(&g);
    cudaFree(&vtest_dev);

    return 0;
}