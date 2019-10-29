#include <iostream>
#include <fstream>

#include "../h/MT.h"

#include "../h/box.cuh"
#include "../h/parameters.cuh"

uint IDs;
uint IDe;
float tau;
float Tfin;

using namespace PhysPeach;
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

    float x1[D*NP];
    float x2[D*NP];
    float v1[D*NP];
    float v2[D*NP];
    float force[D*NP];

    Box box;
    makeBox(&box);
    initBox(&box, 0);

    //checkharmonicEvoBox
    culcHarmonicFint2D<<<NB,NT>>>(
        box.g, 
        box.g.refCell_dev, 
        box.g.cell_dev, 
        box.p.force_dev, 
        box.L, 
        box.p.diam_dev, 
        box.p.x_dev
    );
    cudaMemcpy(x1, box.p.x_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(v1, box.p.v_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(force, box.p.force_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);

    vEvoBD<<<NB,NT>>>(box.p.v_dev, box.dt, 0, box.p.force_dev, box.p.rndState_dev);
    cudaMemcpy(v2, box.p.v_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    for(uint i = 0; i < 10; i++){
        std::cout << i << ": " << v2[i] <<" += ";
        std::cout << box.dt*(-v1[i] + force[i])<< std::endl;
    }

    xEvo<<<NB,NT>>>(box.p.x_dev, box.dt, box.L, box.p.v_dev);
    cudaMemcpy(x2, box.p.x_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    for(uint i = 0; i < 10; i++){
        std::cout << i << ": " << x2[i] <<" += ";
        std::cout << box.dt*v2[i] << std::endl;
    }

    killBox(&box);

    return 0;
}