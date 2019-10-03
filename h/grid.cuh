#ifndef GRID_CUH
#define GRID_CUH
#include <cuda.h>
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
    };
    void makeGrid(Grid* grid, float L);
    void killGrid(Grid* grid);
    void makeCellPattern2D(Grid* grid)
}
#endif