#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "../h/MT.h"
#include "../h/generalFuncs.cuh"
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