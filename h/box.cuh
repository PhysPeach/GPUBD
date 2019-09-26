#ifndef BOX_CUH
#define BOX_CUH

#include <iostream>
#include <fstream>
#include <sstream>

#include <math.h>

#include "parameters.cuh"
#include "particles.cuh"

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
    
        //interactions
        uint* refGrid_dev;
    
        //cell list
        uint M;
        uint EpM;//Num of Elements per M
        uint* needUpdate_dev;
        uint* grid_dev;//[M2][EpM] ->[m*EpM + epm]
        float* positionMemory_dev;
    
        //recorder
        std::ofstream positionFile;
        std::ofstream logFile;
    };
    
    //setters and getters
    inline void setdt_T(Box* box, double setdt, float setT){box->dt = setdt;box->T = setT;box->thermalFuctor = sqrt(2 * setT/setdt);return;}//
    
    //inline float getNU(Box* box);

    void makeBox(Box* box);
    void killBox(Box* box);
    //void initBox(Box* box, uint ID);
    
    //cell list
    /*__global__ void updateGrid2D(
        Box* box,
        uint* map_dev, 
        float* x_dev, 
        float* positionMemory_dev);
    */
    //void judgeUpdateGrid();
    
    //interactions
    /*__global__ void culcInteraction2D(
        Box* box,
        float* force_dev,
        uint* map_dev, 
        uint* pattern_dev, 
        float* diam_dev,
        float* x_dev);
    */
    //time evolution
    //inline void tDvlpBD(Box* box);
    
    /*__global__ void culcHarmonicInteraction2D(
        float* force_dev, 
        Box* box,
        uint* map_dev, 
        uint* pattern_dev,
        float* diam_dev, 
        float* x_dev);
    */
    //inline void tHarmonicDvlp(Box* box);
    
    //equilibrations
    //void coolSys(double Ttmp, double tcool);
    //void equilibrateSys(double teq);
    
    //record
    //void recordSys(Box* box);
    //void getData(Box* box);
}

#endif