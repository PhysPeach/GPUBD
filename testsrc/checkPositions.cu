#include <iostream>
#include <fstream>

#include "../h/MT.h"

#include "../h/particles.cuh"
#include "../h/box.cuh"
#include "../h/parameters.cuh"

uint IDs;
uint IDe;
double tmax;
float Tfin;

using namespace PhysPeach;
int main(){
    std::cout << "hello, test" << std::endl;

    //test
    Tfin = 1;
    char timescale = 7;
    IDs = 0;
    IDe = 0;

    tmax = 1;
    for(char ts = 0; ts <timescale; ts++){
        tmax *= 2;
    }

    //initialise random func
    init_genrand((unsigned long)time(NULL));

    std::cout << "---Settings---" << std::endl;
    std::cout << "Tfin = " << Tfin << std::endl;
    std::cout << "tmax = " << tmax << std::endl;
    std::cout << "ID = [" << IDs << ", " << IDe << "]" << std::endl;
    std::cout << "--------------" << std::endl;

    float force[D*NP];

    Box box;
    makeBox(&box);
    initBox(&box, 0);

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
    
    std::cout << "K:" << K(&box.p) << ", U:" << U(&box.g, box.p.diam_dev, box.p.x_dev) << std::endl;

    std::ofstream checkPositions("testData/checkPositions.data");

    cudaMemcpy(box.p.x, box.p.x_dev, D * NP * sizeof(float), cudaMemcpyDeviceToHost);

    for(uint n = 0; n < NP; n++){
        for(char d = 0; d < D; d++){
            checkPositions << box.p.x[d * NP + n] << " ";
        }
        checkPositions << std::endl;
    }

    checkPositions.close();

    killBox(&box);

    return 0;
}