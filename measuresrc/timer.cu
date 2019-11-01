#include <iostream>
#include <fstream>

#include "../h/timer.cuh"

#include "../h/box.cuh"
#include "../h/parameters.cuh"

uint IDs;
uint IDe;
double tmax;
float Tfin;

using namespace PhysPeach;
int main(){
    std::cout << "hello, evaluate" << std::endl;

    Tfin = 1;
    tmax = 0.1;
    IDs = 0;
    IDe = 0;

    std::cout << "---Settings---" << std::endl;
    std::cout << "NP = " << NP << std::endl;
    std::cout << "NB = " << NB << std::endl;
    std::cout << "NT = " << NT << std::endl;
    std::cout << "IB = " << IB << std::endl;
    std::cout << "IT = " << IT << std::endl;
    std::cout << "--------------" << std::endl;

    Box box;
    makeBox(&box);
    initBox(&box, 0);

    //Routine
    const uint loop = 10000;
    double endTime;
    std::cout << "starting benchmark" << std::endl;
    measureTime();
    benchmark(&box, loop);
    endTime = measureTime();

    std::cout << "---Results---" << std::endl;
    std::cout << "Time: " << endTime << "ms" << std::endl;
    std::cout << "Loop: " << loop << "steps" << std::endl;
    std::cout << "   -> " << endTime/(double)loop << "ms/step" << std::endl;
    std::cout << "-------------" << std::endl;

    killBox(&box);

    return 0;
}