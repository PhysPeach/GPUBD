#include <iostream>
#include <fstream>

#include "../h/MT.h"

#include "../h/particles.cuh"
#include "../h/parameters.cuh"

uint IDs;
uint IDe;
float tmax;
float Tfin;

using namespace PhysPeach;

__global__ void setVec(float* x, float a){
    uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint i = i_global; i < D * NP; i += NB * NT){
        x[i] = i;
    }
}

int main(){
    std::cout << "hello, test" << std::endl;

    //test
    Tfin = 1;
    tmax = 100;
    IDs = 0;
    IDe = 0;

    //initialise random func
    init_genrand((unsigned long)time(NULL));

    float v1[D*NP];
    float v2[D*NP];

    Particles p;
    makeParticles(&p);
    setVec<<<NB,NT>>>(p.v_dev,6);
    cudaMemcpy(v1, p.v_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    removevg2D(&p);
    cudaMemcpy(v2, p.v_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    killParticles(&p);
    for(uint i = 0; i < NP; i++){
        std::cout << i << ": " << v1[i] <<" -> v1 - vg(~=NP/2) = " << v2[i] << std::endl;
    }

    return 0;
}