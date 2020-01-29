#include <iostream>

#include "../h/box.cuh"
#include "../h/parameters.cuh"

uint IDs;
uint IDe;
double tmax;
float Tfin;

using namespace PhysPeach;
int main(int argc, char** argv){
    IDs = atoi(argv[1]);
	IDe = atoi(argv[2]);
	Tfin = atof(argv[3]);
	char timescale = atoi(argv[4]);

    tmax = 1;
    for(char ts = 0; ts <timescale; ts++){
        tmax *= 2;
    }

    std::cout << "---Settings---" << std::endl;
    std::cout << "N = " << NP << std::endl;
    std::cout << "ID = [" << IDs << ", " << IDe << "]" << std::endl;
    std::cout << "Tfin = " << Tfin << std::endl;
    std::cout << "tmax = " << tmax << std::endl;
    std::cout << "--------------" << std::endl;

    Box box;
    makeBox(&box);
    for(uint i = IDs; i <= IDe; i++){
        initBox(&box, i);
        getDataLD(&box);
        //connectLDtoMD
        //getData(&box);
    }
    killBox(&box);

    return 0;
}