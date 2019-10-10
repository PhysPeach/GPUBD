#include "../h/grid.cuh"

namespace PhysPeach{
    void makeGrid(Grid* grid, float L){
        //define M ~ L/Rcell: Rcell ~ 5a
        grid->M = (uint)(L/(4.8*a0));
        //for small system
        if(grid->M < 3){
            grid->M = 3;
        }
        grid->rc = L/(float)grid->M;
        uint M2 = grid->M * grid->M;
        grid->EpM = (uint)(1.5 * (float)NP /(float)M2); //EpM ~ NP/M^D
        grid->updateFreq = 1;

        cudaMalloc((void**)&grid->cell_dev, M2 * grid->EpM * sizeof(uint));

        //for parallel interactions
        uint M_NG = grid->M/NG + 0.9;
        cudaMalloc((void**)&grid->refCell_dev, M_NG * M_NG * sizeof(uint));
        makeCellPattern2D(grid);

        //for determine updateFreq
        cudaMalloc((void**)&grid->vmax_dev[0], D * NP * sizeof(float));
        cudaMalloc((void**)&grid->vmax_dev[1], D * NP * sizeof(float));

        return;
    }
    void killGrid(Grid* grid){
        cudaFree(grid->refCell_dev);
        cudaFree(grid->cell_dev);
        cudaFree(grid->vmax_dev[0]);
        cudaFree(grid->vmax_dev[1]);
        return;
    }
    void makeCellPattern2D(Grid* grid){
        uint M_NG = grid->M/NG + 0.9;
        uint pattern[M_NG*M_NG];

        uint foo = 0;
        uint hoge = 0;
        uint hhoge = 0;
        for(uint m_NG2 = 0; m_NG2 < M_NG*M_NG; m_NG2++){
            hhoge = m_NG2/M_NG;
            hoge = m_NG2 - hhoge * M_NG;
            if(!hoge){
                foo = NG * hhoge * grid->M;
            }
            pattern[m_NG2] = foo;
            foo += NG;
        }
        cudaMemcpy(grid->refCell_dev, pattern, M_NG * M_NG * sizeof(uint), cudaMemcpyHostToDevice);
        return;
    }
    __global__ void updateGrid2D(Grid grid, uint* cell, float* x){
        uint n_global = blockIdx.x * blockDim.x + threadIdx.x;

        uint M = grid.M;
        uint EpM = grid.EpM;
        float rc = grid.rc;
    
        uint cellPos[D];
        uint cellAddress;//[0, M * M - 1]
        uint n_m;
        uint counter;

        for(uint n = n_global; n < NP; n += NB * NT){
            cellPos[0] = (uint)(x[n]/rc);
            cellPos[1] = (uint)(x[NP+n]/rc);
            cellAddress = cellPos[1] * M + cellPos[0];
            n_m = cellAddress * EpM;
            counter = 1 + atomicAdd(&cell[n_m], 1);
            cell[n_m + counter] = n;
        }
    }
    void setUpdateFreq(Grid* grid, double dt, float *v){
        float vmax;
        uint flip = 0;
        uint l = D * NP;
        reductionMax<<<NB,NT>>>(grid->vmax_dev[0], v, l);
        l = (l + NT-1)/NT;
        while(l > 1){
            flip = !flip;
            reductionMax<<<NB,NT>>>(grid->vmax_dev[flip], grid->vmax_dev[!flip], l);
            l = (l + NT-1)/NT;
        }
        cudaMemcpy(&vmax, grid->vmax_dev[flip], sizeof(float), cudaMemcpyDeviceToHost);
        if(vmax > dt){
            grid->updateFreq = a0/(vmax * dt);
        }
        else{
            grid->updateFreq = 1;
        }
        return;
    }
    inline void checkUpdate(Grid* grid, double dt, float* x, float* v){
        static uint counter = 0;
        counter++;
        if(counter >= grid->updateFreq){
            updateGrid2D<<<NB,NT>>>(*grid, grid->cell_dev, x);
            setUpdateFreq(grid, dt, v);
        }
        return;
    }
}