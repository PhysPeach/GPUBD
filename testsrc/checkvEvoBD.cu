#include <iostream>
#include <fstream>

#include "../h/MT.h"

#include "../h/particles.cuh"
#include "../h/parameters.cuh"

uint IDs;
uint IDe;
float tau;
float Tfin;

using namespace PhysPeach;

__global__ void setVec(float* x, float a){
    uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint i = i_global; i < NP; i += NB * NT){
        x[i] = a;
        x[NP+i] = 2*a;
    }
}

int main(){
    std::cout << "hello, test" << std::endl;

    //test
    Tfin = 1;
    tau = 100;
    IDs = 0;
    IDe = 0;

    //initialise random func
    init_genrand((unsigned long)time(NULL));

    std::cout << "---Settings---" << std::endl;
    std::cout << "Tfin = " << Tfin << std::endl;
    std::cout << "t_eq = " << tau << std::endl;
    std::cout << "t_rec = " << tau << std::endl;
    std::cout << "ID = [" << IDs << ", " << IDe << "]" << std::endl;
    std::cout << "--------------" << std::endl;

    float v1[D*NP];
    float v2[D*NP];
    float force[D*NP];
    double dt = 0.001;

    Particles p;
    makeParticles(&p);
    setVec<<<NB,NT>>>(p.v_dev,5);
    setVec<<<NB,NT>>>(p.force_dev,7);
    cudaMemcpy(v1, p.v_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(force, p.force_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    vEvoBD<<<NB,NT>>>(p.v_dev, dt, 0, p.force_dev, p.rndState_dev);
    cudaMemcpy(v2, p.v_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    killParticles(&p);
    for(uint i = 0; i < D*NP; i++){
        std::cout << i << ": " << v1[i] + dt*(-v1[i] + force[i]) <<" = ";
        std::cout << v2[i] << std::endl;
    }

    return 0;
}