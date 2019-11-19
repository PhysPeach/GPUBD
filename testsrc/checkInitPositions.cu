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
int main(){
    std::cout << "hello, test" << std::endl;

    //test
    Tfin = 1;
    tmax = 1;
    IDs = 0;
    IDe = 0;

    //initialise random func
    init_genrand((unsigned long)time(NULL));

    Particles p;
    makeParticles(&p);
    scatterParticles(&p, 10.0);

    //test initParticles
    std::ofstream checkScatterPositions("testData/checkScatterParticles.data");
    
    cudaMemcpy(p.diam, p.diam_dev, NP * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.x, p.x_dev, D * NP * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.v, p.v_dev, D * NP * sizeof(float), cudaMemcpyDeviceToHost);

    for(uint n = 0; n < NP; n++){
        checkScatterPositions << p.diam[n] << " ";
    }
    checkScatterPositions << std::endl;

    for(uint n = 0; n < NP; n++){
        for(char d = 0; d < D; d++){
            checkScatterPositions << p.x[d * NP + n] << " ";
        }
        for(char d = 0; d < D; d++){
            checkScatterPositions << p.v[d * NP + n] << " ";
        }
        checkScatterPositions << std::endl;
    }

    checkScatterPositions.close();
    
    killParticles(&p);

    std::cout << "makeParticles done!" << std::endl;
    return 0;
}