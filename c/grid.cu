#include "../h/grid.cuh"

namespace PhysPeach{
    void makeGrid(Grid* grid, double L){
        //define M ~ L/Rcell: Rcell ~ 5a
        grid->M = SQRT_NUM_OF_CELLS;
        grid->rc = L/(double)grid->M;
        uint M2 = grid->M * grid->M;
        grid->updateFreq = 1;

        //for parallel interactions
        uint M_NGx = (double)grid->M/(double)NGx + 0.9;
        uint M_NGy = (double)grid->M/(double)NGy + 0.9;
        cudaMalloc((void**)&grid->refCell_dev, M_NGy * M_NGx * sizeof(uint));
        makeCellPattern2D(grid);

        grid->EpM = EPM;
        cudaMalloc((void**)&grid->cell_dev, M2 * grid->EpM * sizeof(uint));

        //for determine updateFreq
        cudaMalloc((void**)&grid->vmax_dev[0], D * NP * sizeof(float));
        cudaMalloc((void**)&grid->vmax_dev[1], D * NP * sizeof(float));

        //for setters and getters
        cudaMalloc((void**)&grid->getNU_dev[0], NP * sizeof(float));
        cudaMalloc((void**)&grid->getNU_dev[1], NP * sizeof(float));

        return;
    }
    void killGrid(Grid* grid){
        cudaFree(grid->refCell_dev);
        cudaFree(grid->cell_dev);
        cudaFree(grid->vmax_dev[0]);
        cudaFree(grid->vmax_dev[1]);
        cudaFree(grid->getNU_dev[0]);
        cudaFree(grid->getNU_dev[1]);
        return;
    }
    void makeCellPattern2D(Grid* grid){
        uint M_NGx = (double)grid->M/(double)NGx + 0.9;
        uint M_NGy = (double)grid->M/(double)NGy + 0.9;
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
    __global__ void updateGrid2D(Grid grid, uint* cell, double* x){
        uint n_global = blockIdx.x * blockDim.x + threadIdx.x;

        uint M = grid.M;
        uint EpM = grid.EpM;
        double rc = grid.rc;
    
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
    void checkUpdate(Grid* grid, double dt, double* x, float* v){
        static uint counter = 0;
        counter++;
        if(counter >= grid->updateFreq){
            setIntVecZero<<<NB,NT>>>(grid->cell_dev, grid->M * grid->M * grid->EpM);
            updateGrid2D<<<NB,NT>>>(*grid, grid->cell_dev, x);
            setUpdateFreq(grid, dt, v);
            counter = 0;
        }
        return;
    }
    __global__ void culcHarmonicFint2D(
        Grid grid, 
        uint *refCell, 
        uint *cell, 
        float *force, 
        double L, 
        float *diam, 
        double *x
    ){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        const int EpM = EPM;
        double rc = grid.rc;
        int M = grid.M;

        //for cells
        int cellPosBasis[D];
        int cellPos[D], cellAddress;

        int nm, NofP;

        //for Fint
        uint j;
        double Lh = 0.5 * L;
        float f_rij;
        double xij[D];
        float rij2, aij, aij2;

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
                            xij[0] = x[j] - x[i];
                            xij[1] = x[NP+j] - x[NP+i];
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
    __global__ void culcFint2D(
        Grid grid, 
        uint *refCell, 
        uint *cell, 
        float *force, 
        double L, 
        float *diam, 
        double *x
    ){
        uint i_block = blockIdx.x;
        uint i_local = threadIdx.x;
        //debug
        uint i_global = i_block * blockDim.x + i_local;

        //save on resister
        const int EpM = EPM;
        int M = grid.M;
        const uint Mx_l = NGx + 2;
        const uint My_l = NGy + 2;
        const uint Mxy_l = My_l * Mx_l;

        //0 <= i_subblock < NGx*NGy, 0 <= i_sublocal < EpM
        uint i_subblock = i_local/EpM;
        uint i_sublocal = i_local - i_subblock * EpM;
        //i_subblock == NGx*NGy must be ignored

        //local
        __shared__ char active[Mxy_l][EpM];
        __shared__ uint cell_s[Mxy_l][EpM];
        __shared__ float diam_s[Mxy_l][EpM];
        __shared__ double x_s[D][Mxy_l][EpM];
        int cellPos_l[D];//0 <=cellPos_l < M_l
        int cellAddress_l;

        //for cells
        int cellPosBasis[D], cellAddressBasis;
        int cellPos[D], cellAddress;

        int nm, nml, NofP;

        //for Fint
        double Lh = 0.5 * L;
        float fi[D], f_rij;
        double xij[D];
        float rij2, aij2, ar2, ar6;

        //while: cellAddress < M*M
        uint M_NGx = (double)M/(double)NGx + 0.9;
        uint M_NGy = (double)M/(double)NGy + 0.9;
        uint blockmax = M_NGx * M_NGy;
        for(uint ib = i_block; ib < blockmax; ib += IB){
            //determine Cell_Basis
            cellAddressBasis = refCell[ib];
            cellPosBasis[1] = cellAddressBasis / grid.M;
            cellPosBasis[0] = cellAddressBasis - cellPosBasis[1] * M;

            //load centre Mems and sorroundings
            if(i_subblock < NGx*NGy){
                for(uint isb = i_subblock; isb < Mxy_l; isb+=NGx*NGy){
                    cellPos_l[1] = isb/Mx_l;
                    cellPos_l[0] = isb - cellPos_l[1] * Mx_l;
                    //if cellAddress is out of area, active[cellAdd_lcl][0] = 0
                    cellPos[0] = (cellPosBasis[0] + cellPos_l[0]) - 1;
                    cellPos[1] = (cellPosBasis[1] + cellPos_l[1]) - 1;
                    //active[][1<=i_sublocal] are just for avoiding memory contention
                    active[isb][i_sublocal] = 1;
                    if(cellPos[0] < 0){cellPos[0] += M; active[isb][i_sublocal] = 0;} 
                    if(cellPos[0] >= M){cellPos[0] -= M; active[isb][i_sublocal] = 0;}
                    if(cellPos[1] < 0){cellPos[1] += M; active[isb][i_sublocal] = 0;}
                    if(cellPos[1] >= M){cellPos[1] -= M; active[isb][i_sublocal] = 0;}
                    
                    cellAddress = cellPos[1]*M + cellPos[0];
                    nm = cellAddress * EpM;
                    cell_s[isb][i_sublocal] = cell[nm + i_sublocal];
                    //i_sublocal = 0 is meaningless for diam and x
                    diam_s[isb][i_sublocal] = diam[cell_s[isb][i_sublocal]];
                    x_s[0][isb][i_sublocal] = x[cell_s[isb][i_sublocal]];
                    x_s[1][isb][i_sublocal] = x[NP + cell_s[isb][i_sublocal]];
                }
            }
            __syncthreads();
            cellPos_l[1] = i_subblock/NGx;
            cellPos_l[0] = i_subblock - cellPos_l[1]*NGx + 1;
            cellPos_l[1]++;
            cellAddress_l = cellPos_l[1] * Mx_l + cellPos_l[0];

            //parallel processing by i_sublocal
            if(i_subblock < NGx*NGy && active[cellAddress_l][0] && 1 <= i_sublocal && i_sublocal <= cell_s[cellAddress_l][0]){
                //culc Interactions
                fi[0] = 0.; fi[1] = 0.;
                for(int mlx = -1; mlx <= 1; mlx++){
                    for(int mly = -1; mly <= 1; mly++){
                        nml = cellAddress_l + mly*Mx_l + mlx;
                        NofP = cell_s[nml][0];
                        for(uint j = 1; j <= NofP; j++){
                            //avoid interaction between same particles
                            if(!(cellAddress_l == nml && i_sublocal == j)){
                                xij[0] = x_s[0][nml][j] - x_s[0][cellAddress_l][i_sublocal];
                                xij[1] = x_s[1][nml][j] - x_s[1][cellAddress_l][i_sublocal];
                                if(xij[0] > Lh){xij[0] -= L;}
                                if(xij[1] > Lh){xij[1] -= L;}
                                if(xij[0] < -Lh){xij[0] += L;}
                                if(xij[1] < -Lh){xij[1] += L;}
                                rij2 = xij[0]*xij[0] + xij[1]*xij[1];
                                aij2 = 0.5 * (diam_s[cellAddress_l][i_sublocal] + diam_s[nml][j]);
                                aij2 *= aij2;
                                ar2 = aij2/rij2;
                                if(1 < 9*ar2){
                                    ar6 = ar2 * ar2 * ar2;
                                    f_rij = -12 * (ar6*ar6)/rij2;
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
    __global__ void culcUint2D(
        Grid grid, 
        uint *refCell, 
        uint *cell, 
        float *U, 
        double L, 
        float *diam, 
        double *x
    ){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        const int EpM = EPM;
        double rc = grid.rc;
        int M = grid.M;

        //for cells
        int cellPosBasis[D];
        int cellPos[D], cellAddress;

        int nm, NofP;

        //for Fint
        uint j;
        double Lh = 0.5 * L;
        double xij[D];
        float rij2, aij2, ar2, ar6;
        float C = 1/(3*3*3*3*3*3 * 3*3*3*3*3*3);

        for(uint i = i_global; i < NP; i += NB*NT){
            U[i] = 0;

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
                            xij[0] = x[j] - x[i];
                            xij[1] = x[NP+j] - x[NP+i];
                            if(xij[0] > Lh){xij[0] -= L;}
                            if(xij[1] > Lh){xij[1] -= L;}
                            if(xij[0] < -Lh){xij[0] += L;}
                            if(xij[1] < -Lh){xij[1] += L;}
                            rij2 = xij[0]*xij[0] + xij[1]*xij[1];
                            aij2 = 0.5 * (diam[i] + diam[j]);
                            aij2 *= aij2;
                            ar2 = aij2/rij2;
                            if(1 < 9*ar2){
                                ar6 = ar2 * ar2 * ar2;
                                U[i] += ar6*ar6 - C;
                            }
                        }
                    }
                }
            }
        }
    }
    float U(Grid* grid, float *diam, double *x){
        float NU = 0;
        uint flip = 0;

        culcUint2D<<<NB,NT>>>(
            *grid,
            grid->refCell_dev,
            grid->cell_dev,
            grid->getNU_dev[0],
            grid->M * grid->rc,
            diam,
            x
        );
        //summations
        for(uint l = NP; l > 1; l = (l + NT-1)/NT){
            flip = !flip;
            reductionSum<<<NB,NT>>>(grid->getNU_dev[flip], grid->getNU_dev[!flip], l);
        }
        cudaMemcpy(&NU, grid->getNU_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);

        return NU/NP;
    }
}