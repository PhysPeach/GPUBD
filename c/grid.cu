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
    __global__ void culcHarmonicInteractions2D(
        Grid grid, 
        uint *refCell, 
        uint *cell, 
        float *force, 
        float L, 
        float *diam, 
        float *x
    ){
        uint i_block = blockIdx.x;
        uint i_local = threadIdx.x;
    
        //save on resister
        const uint EpM = EPM;
        uint M = grid.M;
        const uint Mx_l = NGx + 2;
        const uint My_l = NGy + 2;
        const uint Mxy_l = My_l * Mx_l;
    
        //0 <= i_subblock < NGx*NGy, 0 <= i_sublocal < EpM
        uint i_subblock = i_local/EpM;
        uint i_sublocal = i_local - i_subblock * EpM;
    
        //local
        uint active[Mxy_l];
        __shared__ unsigned int cell_s[Mxy_l][EpM];
        __shared__ float diam_s[Mxy_l][EpM];
        __shared__ float x_s[D][Mxy_l][EpM];
        uint cellPos_l[D];//0 <=cellPos_l < M_l
        uint cellAddress_l;
    
        uint cellPos[D], cellAddress;
    
        uint cellPosBasis[D], cellAddressBasis;
        uint nm, nml, NofP;
    
        //for interactions
        float Lh = 0.5 * L;
    
        float fi[D], f_rij;
        float xij[D], rij2, aij, aij2;
    
        //while: cellAddress < M*M
        uint M_NGx = (float)M/(float)NGx + 0.9;
        uint M_NGy = (float)M/(float)NGy + 0.9;
        uint blockmax = M_NGx * M_NGy;
        for(uint ib = i_block; ib < blockmax; ib += IB){
            cellAddressBasis = refCell[ib];
            cellPosBasis[1] = cellAddressBasis / grid.M;
            cellPosBasis[0] = cellAddressBasis - cellPosBasis[1] * M;
    
            //load centre Mems and sorroundings
            for(uint isb = i_subblock; isb < Mxy_l; isb+=NGx*NGy){
                cellPos_l[1] = isb/Mx_l;
                cellPos_l[0] = isb - cellPos_l[1]*Mx_l;
                //if cellAddress is out of area, active[cellAdd_lcl][0] = 0
                cellPos[0] = checkActiveCell(&active[isb], cellPosBasis[0]-1 + cellPos_l[0], M);
                cellPos[1] = checkActiveCell(&active[isb], cellPosBasis[1]-1 + cellPos_l[1], M);
                cellAddress = cellPos[1]*M + cellPos[0];
                nm = cellAddress * EpM;
                cell_s[isb][i_sublocal] = cell[nm + i_sublocal];
                //i_sublocal = 0 is meaningless for diam and x
                diam_s[isb][i_sublocal] = diam[cell_s[isb][i_sublocal]];
                x_s[0][isb][i_sublocal] = x[cell_s[isb][i_sublocal]];
                x_s[1][isb][i_sublocal] = x[NP + cell_s[isb][i_sublocal]];
            }
            __syncthreads();
            cellPos_l[1] = i_subblock/Mx_l;
            cellPos_l[0] = i_subblock - cellPos_l[1]*Mx_l + 1;
            cellPos_l[1]++;
            cellAddress_l = cellPos_l[1] * Mx_l + cellPos_l[0];
            if(active[cellAddress_l]==1 && i_sublocal != 0 && i_sublocal <= cell_s[cellAddress_l][0]){
                //culc Interactions
                fi[0] = 0.; fi[1] = 0.;
                for(int mlx = -1; mlx <= 1; mlx++){
                    for(int mly = -1;mly <= mly; mly++){
                        nml = cellAddress_l + mly*Mx_l + mlx;
                        NofP = cell_s[nml][0];
                        for(uint j = 1; j <= NofP; j++){
                            //avoid interaction between same particles
                            if(cellAddress_l != nml || i_sublocal != j){
                                xij[0] = x_s[0][cellAddress_l][i_sublocal] - x_s[0][nml][j];
                                xij[1] = x_s[1][cellAddress_l][i_sublocal] - x_s[1][nml][j];
                                if(xij[0] > Lh){xij[0] -= L;}
                                if(xij[1] > Lh){xij[1] -= L;}
                                if(xij[0] < -Lh){xij[0] += L;}
                                if(xij[1] < -Lh){xij[1] += L;}
                                rij2 = xij[0]*xij[0] + xij[1]*xij[1];
                                aij = 0.5 * (diam_s[cellAddress_l][i_sublocal] + diam_s[nml][j]);
                                aij2 = aij * aij;
                                if(rij2 < aij2){
                                    f_rij = 50 * (1/aij2 - 1/(aij*sqrt(rij2)));
                                    fi[0] += f_rij * xij[0];
                                    fi[1] += f_rij * xij[1];
                                }
                            }
                        }
                    }
                }
                //end culc Interactions
                //index = cell_s[cellAddress_l][i_sublocal]
                force[cell_s[cellAddress_l][i_sublocal]] = fi[0];
                force[NP + cell_s[cellAddress_l][i_sublocal]] = fi[1];
            }
        }
    }
}