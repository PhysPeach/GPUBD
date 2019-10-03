#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <assert.h>

#include "../h/MT.h"
#include "../h/grid.cuh"

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

    std::cout << "checkGridCreate" << std::endl;

    Grid g;
    float L = 35.9;
    makeGrid(&g, L);
    std::cout << L << " " << g.M * g.rc << std::endl;
    std::cout << g.EpM << std::endl;
    killGrid(&g);

    std::cout << "test done!" << std::endl;
    return 0;
}