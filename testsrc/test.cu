#include <iostream>
#include <fstream>

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

    std::cout << "---Settings---" << std::endl;
    std::cout << "Tfin = " << Tfin << std::endl;
    std::cout << "tmax = " << tmax << std::endl;
    std::cout << "ID = [" << IDs << ", " << IDe << "]" << std::endl;
    std::cout << "--------------" << std::endl;

    Box box;
    makeBox(&box);
    for(uint i = IDs; i <= IDe; i++){
        initBox(&box, i);
        getData(&box);
    }
    killBox(&box);

    return 0;
}