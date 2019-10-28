#include "../h/grid.cuh"

namespace PhysPeach{
    __device__ int checkActiveCell(uint* active, int a, int b){
        *active = 1;
        if(a < 0){
            *active = 0;
            a += b;
        }
        if(a >= b){
            *active = 0;
            a -= b;
        }
        return a;
    }
    void makeGrid(Grid* grid, float L){
        //define M ~ L/Rcell: Rcell ~ 5a
        grid->M = SQRT_NUM_OF_CELLS;
        grid->rc = L/(float)grid->M;
        uint M2 = grid->M * grid->M;
        grid->updateFreq = 1;

        //for parallel interactions
        uint M_NGx = (float)grid->M/(float)NGx + 0.9;
        uint M_NGy = (float)grid->M/(float)NGy + 0.9;
        cudaMalloc((void**)&grid->refCell_dev, M_NGy * M_NGx * sizeof(uint));
        makeCellPattern2D(grid);

        grid->EpM = EPM;
        cudaMalloc((void**)&grid->cell_dev, M2 * grid->EpM * sizeof(uint));

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
        uint M_NGx = (float)grid->M/(float)NGx + 0.9;
        uint M_NGy = (float)grid->M/(float)NGy + 0.9;
        uint pattern[M_NGy*M_NGx];

        uint Pos[D];
        for(uint i = 0; i < M_NGx*M_NGy; i++){
            Pos[1] = i / M_NGx;
            Pos[0] = i - Pos[1] * M_NGx;

            pattern[i] = Pos[1] * NGy * grid->M + Pos[0] * NGx;
        }
        cudaMemcpy(grid->refCell_dev, pattern, M_NGx * M_NGy * sizeof(uint), cudaMemcpyHostToDevice);
        return;
    }
    __global__ void updateGrid2D(Grid grid, uint* cell, float* x){
        uint n_global = blockIdx.x * blockDim.x + threadIdx.x;

        uint M = grid.M;
        uint EpM = grid.EpM;
        float rc = grid.rc;
    
        uint cellPosBasis[D];
        uint cellAddress;//[0, M * M - 1]
        uint n_m;
        uint counter;

        for(uint n = n_global; n < NP; n += NB * NT){
            cellPosBasis[0] = (uint)(x[n]/rc);
            cellPosBasis[1] = (uint)(x[NP+n]/rc);
            cellAddress = cellPosBasis[1] * M + cellPosBasis[0];
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
    void checkUpdate(Grid* grid, double dt, float* x, float* v){
        static uint counter = 0;
        counter++;
        if(counter >= grid->updateFreq){
            setIntVecZero<<<NB,NT>>>(grid->cell_dev, grid->M * grid->M * grid->EpM);
            updateGrid2D<<<NB,NT>>>(*grid, grid->cell_dev, x);
            setUpdateFreq(grid, dt, v);
        }
        return;
    }
    __global__ void culcHarmonicFint2D(
        Grid grid, 
        uint *refCell, 
        uint *cell, 
        float *force, 
        float L, 
        float *diam, 
        float *x
    ){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        const int EpM = EPM;
        float rc = grid.rc;
        int M = grid.M;

        //for cells
        int cellPosBasis[D];
        int cellPos[D], cellAddress;

        int nm, NofP;

        //for Fint
        uint j;
        float Lh = 0.5 * L;
        float f_rij;
        float xij[D], rij2, aij, aij2;

        for(uint i = i_global; i < NP; i += NB*NT){
            force[i] = 0; force[i+NP]=0;

            cellPosBasis[0] = x[i]/rc;
            cellPosBasis[1] = x[i+NP]/rc;
            if(cellPosBasis[0] == -1) cellPosBasis[0] = M - 1;
            if(cellPosBasis[0] == M) cellPosBasis[0] = 0;
            if(cellPosBasis[1] == -1) cellPosBasis[1] = M - 1;
            if(cellPosBasis[1] == M) cellPosBasis[1] = 0;
            
            for(int mlx = -1; mlx <= 1; mlx++){
                cellPos[0] = cellPosBasis[0] + mlx;
                if(cellPos[0] == -1) cellPos[0] = M - 1;
                if(cellPos[0] == M) cellPos[0] = 0;
                for(int mly = -1; mly <= 1; mly++){
                    cellPos[1] = cellPosBasis[1] + mly;
                    if(cellPos[1] == -1) cellPos[1] = M - 1;
                    if(cellPos[1] == M) cellPos[1] = 0;
                    cellAddress = cellPos[1] * M + cellPos[0];
                    nm = cellAddress * EpM;
                    NofP = cell[nm];//1 <= k <= NofP

                    for(uint k = 1; k <=NofP; k++){
                        j = cell[nm+k];
                        if(i!=j){
                            xij[0] = x[i] - x[j];
                            xij[1] = x[NP+i] - x[NP+j];
                            if(xij[0] > Lh){xij[0] -= L;}
                            if(xij[1] > Lh){xij[1] -= L;}
                            if(xij[0] < -Lh){xij[0] += L;}
                            if(xij[1] < -Lh){xij[1] += L;}
                            rij2 = xij[0]*xij[0] + xij[1]*xij[1];
                            aij = 0.5 * (diam[i] + diam[j]);
                            aij2 = aij * aij;
                            if(rij2 < aij2){
                                f_rij = 50 * (1/aij2 - 1/(aij*sqrt(rij2)));
                                force[i] += f_rij * xij[0];
                                force[i+NP] += f_rij * xij[1];
                            }
                        }
                    }
                }
            }
        }
    }
}