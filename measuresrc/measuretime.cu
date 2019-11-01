#include <iostream>
#include <fstream>
#include <chrono>

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
    tmax = 1;
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

    std::chrono::system_clock::time_point start_time, end_time;
    std::cout << "start" << std::endl;
	start = std::chrono::system_clock::now();

	//Routine
	for (int loop = 0; loop < 1000000; loop++) {
		tEvoBox(&box);
    }
	end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << elapsed << "ms" << std::endl;
    killBox(&box);

    return 0;
}