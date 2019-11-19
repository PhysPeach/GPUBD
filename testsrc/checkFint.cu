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

    float force[D*NP];

    Box box;
    makeBox(&box);
    initBox(&box, 1);

    float av = 0;
    float sig = 0.;
    cudaMemcpy(force, box.p.force_dev, D*NP*sizeof(float),cudaMemcpyDeviceToHost);
    for(uint i = 0; i < D*NP; i++){
        std::cout << i << ": " << force[i] << std::endl;
        av += force[i]/(D*NP);
    }
    for(uint i = 0; i < D*NP; i++){
        sig += (av - force[i])*(av - force[i])/(D*NP);
    }
    sig = sqrt(sig);
    std::cout <<"force: av = " << av << ", sig = " << sig << std::endl;

    killBox(&box);

    return 0;
}