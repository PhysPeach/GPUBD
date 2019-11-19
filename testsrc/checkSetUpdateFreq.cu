#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "../h/grid.cuh"
#include "../h/parameters.cuh"

unsigned int IDs;
unsigned int IDe;
float tmax;
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
    tmax = 1;
    IDs = 0;
    IDe = 0;

    //initialise random func
    init_genrand((unsigned long)time(NULL));

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