#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "../h/generalFuncs.cuh"
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

    float x;
    uint flip = 0;
    float *x_dev[2];
    cudaMalloc((void**)&x_dev[0], D * NP * sizeof(uint));
    cudaMalloc((void**)&x_dev[1], D * NP * sizeof(uint));

    //x[NP] = 50, else 1
    setVec<<<NB,NT>>>(x_dev[0]);
    for(uint l = D * NP; l > 1; l = (l+NT-1)/NT){
        flip = !flip;
        reductionMax<<<NB,NT>>>(x_dev[flip], x_dev[!flip], l);
    }
    cudaMemcpy(&x, x_dev[flip],  sizeof(float), cudaMemcpyDeviceToHost);
    printf("N = %d, 50. = %f\n", D * NP, x);

    cudaFree(&x_dev[0]);
    cudaFree(&x_dev[1]);

    return 0;
}