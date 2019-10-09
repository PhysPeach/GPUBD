#ifndef GRID_CUH
#define GRID_CUH
#include <cuda.h>
#include "generalFuncs.cuh"
#include "parameters.cuh"
namespace PhysPeach{
    struct Grid{
        //reference
        uint* refCell_dev;

        //cell list
        float rc;
        uint M;
        uint EpM;//(Num of Elements per M) - 1
        uint* cell_dev;//[M2][EpM] ->[m2* EpM + epm] epm = 0->NofE
        uint updateFreq;
        float* vmax_dev[2];
    };
    void makeGrid(Grid* grid, float L);
    void killGrid(Grid* grid);
    void makeCellPattern2D(Grid* grid);
    __global__ void updateGrid2D(Grid grid, uint* cell, float* x);
    void setUpdateFreq(Grid* grid, double dt, float *v);
    inline void checkUpdate(Grid* grid, double dt, float* x, float* v);
}
#endif