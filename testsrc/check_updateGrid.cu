#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "../h/box.cuh"
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

    Box box;
    makeBox(&box);
    initBox(&box,1);
    //checkUpdateGrid
    std::cout << "check_updateGrid" << std::endl;
    std::ofstream check_updateGrid("testData/check_updateGrid.data");

    uint *grid;
    uint M2 = box.M * box.M;
    uint bEpM = box.EpM;
    std::cout << "M2 = " << M2 << std::endl;
    std::cout << "EpM = " << bEpM << std::endl;
    grid = (uint*)malloc(M2 * bEpM * sizeof(uint));
    cudaMemcpy(cell, box.g.cell_dev, M2 * bEpM * sizeof(uint), cudaMemcpyDeviceToHost);
    for(uint i = 0; i < M2; i++){
        for(uint c = 0; c < bEpM; c++){
            check_updateGrid << "i: " << i << ", c: " << c << " , " << cell[i*box.EpM+c] << std::endl;
        }
        check_updateGrid << std::endl;
    }

    free(cell);
    check_updateGrid.close();

    killBox(&box);

    std::cout << "test done!" << std::endl;
    return 0;
}