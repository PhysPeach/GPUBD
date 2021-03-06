#ifndef GRID_CUH
#define GRID_CUH
#include <cuda.h>
#include <iostream>
#include "generalFuncs.cuh"
#include "parameters.cuh"
namespace PhysPeach{
    struct Grid{
        //reference
        uint* refCell_dev;

        //cell list
        double rc;
        uint M;
        uint EpM;//(Num of Elements per M) - 1
        uint* cell_dev;//[M2][EpM] ->[m2* EpM + epm] epm = 0->NofE
        uint updateFreq;
        float* vmax_dev[2];

        //for setters and getters
        float *getNU_dev[2]; //U[N]
    };
    void makeGrid(Grid* grid, double L);
    void killGrid(Grid* grid);
    void makeCellPattern2D(Grid* grid);
    __global__ void updateGrid2D(Grid grid, uint* cell, double* x);
    void setUpdateFreq(Grid* grid, double dt, float *v);
    void checkUpdate(Grid* grid, double dt, double* x, float* v);
    __global__ void culcHarmonicFint2D(
        Grid g, 
        uint *refCell, 
        uint *cell, 
        float *force, 
        float *diam, 
        double *x);
    __global__ void culcFint2D(
        Grid g,
        uint *refCell, 
        uint *cell, 
        float *force, 
        float *diam, 
        double *x);
    
    //for setters and getters
    float U(Grid* grid, float *diam, double *x);
}

#endif