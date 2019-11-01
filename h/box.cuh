#ifndef BOX_CUH
#define BOX_CUH

#include <iostream>
#include <fstream>
#include <sstream>

#include <math.h>

#include "generalFuncs.cuh"
#include "parameters.cuh"
#include "particles.cuh"
#include "grid.cuh"

namespace PhysPeach{
    struct Box{
        //Declare System
        uint id;
        double dt;
        double t; //time
        float T; //temparature
        float thermalFuctor; //sqrt(2*ZT * T /dt)
        float L; //length of box
        Particles p;
    
        //cell list
        Grid g;
    
        //recorder
        std::ofstream positionFile;
        std::ofstream animeFile;
    };
    
    //setters and getters
    inline void setdt_T(Box* box, double setdt, float setT){box->dt = setdt;box->T = setT;box->thermalFuctor = sqrt(2 * setT/setdt);return;}//
    
    //inline float getNU(Box* box);

    void makeBox(Box* box);
    void killBox(Box* box);
    void initBox(Box* box, uint ID);

    //time evolution
    inline void harmonicEvoBox(Box* box);
    inline void tEvoBox(Box* box);
    
    //equilibrations
    void equilibrateBox(Box* box, double teq);
    
    //record
    void recBox(std::ofstream *of, Box* box);
    void getData(Box* box);
}

#endif