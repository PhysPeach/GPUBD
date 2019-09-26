#include <iostream>

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
    killParticles(&p);

    std::cout << "makeParticles done!" << std::endl;
    return 0;
}