#include <iostream>
#include <fstream>

#include "../h/MT.h"

#include "../h/particles.cuh"
#include "../h/parameters.cuh"

unsigned int IT;
unsigned int IDs;
unsigned int IDe;
float tau;
float Tfin;

using namespace PhysPeach;
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

    Particles p;
    makeParticles(&p);
    initParticles(&p, 10.0);

    //test initParticles
    std::ofstream checkInitPositions("testData/checkInitParticles.data");
    
    cudaMemcpy(p.diam, p.diam_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.x, p.x_dev, D * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.v, p.v_dev, D * N * sizeof(float), cudaMemcpyDeviceToHost);

    for(unsigned int n = 0; n < N; n++){
        checkInitPositions << p.diam[n] << " ";
    }
    checkInitPositions << std::endl;

    for(unsigned int n = 0; n < N; n++){
        for(char d = 0; d < D; d++){
            checkInitPositions << p.x[d * N + n] << " ";
        }
        for(char d = 0; d < D; d++){
            checkInitPositions << p.v[d * N + n] << " ";
        }
        checkInitPositions << std::endl;
    }

    checkInitPositions.close();
    
    killParticles(&p);

    std::cout << "makeParticles done!" << std::endl;
    return 0;
}