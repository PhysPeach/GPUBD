#ifndef BOX_CUH
#define BOX_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "generalFuncs.cuh"
#include "parameters.cuh"
#include "particles.cuh"
#include "grid.cuh"

namespace PhysPeach{
    struct Box{
        //Declare System
        uint id;
        double dt;
        float Tset; //setting temparature
        float Eav; //average of K + U on LD system
        float thermalFuctor; //sqrt(2*ZT * T /dt)
        double L; //length of box
        Particles p;
    
        //cell list
        Grid g;

        //for record
        std::string NTDir;
        std::string LDDir;
        std::string MDDir;
        std::string EDir;
        std::string posDir;
        std::string velDir;
    };
    
    //setters and getters
    inline void setdt_T(Box* box, double setdt, float setT){box->dt = setdt;box->Tset = setT;box->thermalFuctor = sqrt(2 * setT/setdt);return;}//
    
    //inline float getNU(Box* box);
    void prepareBox(Box* box);
    void makeBox(Box* box);
    void killBox(Box* box);
    void initBox(Box* box, uint ID);

    //time evolution
    inline void harmonicEvoBox(Box* box);
    inline void tEvoLD(Box* box);
    inline void tEvoMD(Box* box);
    
    //equilibrations
    void equilibrateBox(Box* box, double teq);
    void connectLDtoMD(Box* box);
    
    //record
    void recPos(std::ofstream *of, Box* box);
    void getDataLD(Box* box);
    void getDataMD(Box* box);
    void benchmark(Box* box, uint loop);
}

#endif