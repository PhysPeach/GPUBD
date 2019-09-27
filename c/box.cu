#include "../h/box.cuh"

namespace PhysPeach{
    void makeGridPattern2D(Box* box){
        uint M_NG = box->M/NG + 0.9;
        uint pattern[M_NG*M_NG];

        uint foo = 0;
        uint hoge = 0;
        uint hhoge = 0;
        for(uint m_NG2 = 0; m_NG2 < M_NG*M_NG; m_NG2++){
            hhoge = m_NG2/M_NG;
            hoge = m_NG2 - hhoge * M_NG;
            if(!hoge){
                foo = NG * hhoge * box->M;
            }
            pattern[m_NG2] = foo;
            foo += NG;
        }
        cudaMemcpy(box->refGrid_dev, pattern, M_NG * M_NG * sizeof(uint), cudaMemcpyHostToDevice);
        return;
    }
    void makeBox(Box* box){

        makeParticles(&box->p);

        //default settings
        box->id = 0;
        box->dt = dt_MD;
        box->t = 0.;
        box->T = Tfin;
        box->L = sqrt(NP/DNSTY);
        box->thermalFuctor = sqrt(2*box->T/box->dt);

        //for list
        ////define M ~ L/Rcell: Rcell ~ 4~5a
        box->M = (uint)(box->L/(4.3*a0));
        ////for small system
        if(box->M < 3){
            box->M = 3;
        }
        uint M2 = box->M * box->M;
        box->EpM = (uint)1.8 * DNSTY * (4.3*a0)*(4.3*a0); //EpM ~ DNSTY * 4.3a0^2 ~ DNSTY * 26.6
        cudaMalloc((void**)&box->needUpdate_dev, sizeof(uint));
        cudaMalloc((void**)&box->positionMemory_dev, D * NP *sizeof(float));
        cudaMalloc((void**)&box->grid_dev, M2 * box->EpM * sizeof(uint));

        //for parallel interactions
        IT = box->EpM * NG * NG;
        uint M_NG = box->M/NG + 0.9;
        cudaMalloc((void**)&box->refGrid_dev, M_NG * M_NG * sizeof(uint));
        makeGridPattern2D(box);

        std::cout << "Made Box" << std::endl;
        
        return;
    }
    void killBox(Box* box){
        //for list
        cudaFree(box->needUpdate_dev);
        cudaFree(box->positionMemory_dev);
        cudaFree(box->grid_dev);

        //for parallel interactions
        cudaFree(box->refGrid_dev);

        //others
        killParticles(&box->p);
        std::cout << "Killed Box" << std::endl;

        return;
    }

    void prepareBox(Box* box){
        box->logFile << "Set InitPositions" << std::endl;
        scatterParticles(&box->p, box->L);
        cudaMemcpy(box->p.diam, box->p.diam_dev, NP * sizeof(float),cudaMemcpyDeviceToHost);
        for(uint n = 0; n < NP; n++){
            box->positionFile << box->p.diam[n] << " ";
        }
        box->positionFile << std::endl << std::endl;
        //set posMem and list
        updateGrid2D<<<NB,NT>>>(box, box->grid_dev, box->positionMemory_dev, box->p.x_dev);
        //remove overraps by using harmonic potential
        uint Nt = 20. / box->dt;
        /*for(int nt = 0; nt < Nt; nt++){
            tHarmonicDvlp();
            judgeUpdateGrid();
        }*/
        
        box->logFile << "-> SIP Done!" << std::endl;
        return;
    }
    void initBox(Box* box, uint ID){
        box->id = ID;
        std::cout << "Start Initialisation: ID = " << box->id << std::endl;
    
       //for record
        std::ostringstream positionFileName;
        positionFileName << "../pos/N" << (uint)NP << "/T" << Tfin << "/posBD_N" << (uint)NP << "_T" << Tfin << "_id" << box->id <<".data";
        box->positionFile.open(positionFileName.str().c_str());
        std::ostringstream logFileName;
        logFileName << "../log/N" << (uint)NP << "/T" << Tfin << "/logBD_N" << (uint)NP << "_T" << Tfin << "_id" << box->id <<".log";
        box->logFile.open(logFileName.str().c_str());

        box->logFile << "Start Initialisation: ID = " << box->id << std::endl;
        box->logFile << "Created Box ID = " << box->id << std::endl;
        
        //hotstart(Tinit >> 1)
        setdt_T(box, dt_INIT, Tinit);

        prepareBox(box);
    
        //equilibrateSys(30.0);
    
        //Tinit -> Tfin
        //coolSys(Tfin, tau);
    
        //setdt_T(dt_BD, Tfin);
    
        //equilibrateSys(10 * tau);
    
        box->logFile << "-> Init Done!" << std::endl;
        return;
    }
    //for grid
    __global__ void updateGrid2D(Box* box, uint* grid, float* positionMemory, float* x){
        uint n_global = blockIdx.x * blockDim.x + threadIdx.x;

        float bL = box->L;
        uint bM = box->M;
        uint bEpM = box->EpM;
        float rc = bL/(float)bM;
    
        uint gridPos[D];
        uint gridAddress;//[0, M * M - 1]
        uint n_m;
        uint counter;
    
        for(uint n = n_global; n < NP; n += NB* NT){
            gridPos[0] = (uint)x[n]/rc;
            gridPos[1] = (uint)x[NP+n]/rc;
            gridAddress = gridPos[1] * bM + gridPos[0];
            n_m = gridAddress * bEpM;
            counter = 1 + atomicAdd(&grid[n_m], 1);
            grid[n_m + counter] = n;
            positionMemory[n] = x[n];
            positionMemory[NP + n] = x[NP + n];
        }
    }

    __global__ void checkGrid(uint* needUpdate, float L, float* x, float* positionMemory){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        uint i_local = threadIdx.x;
    
        __shared__ uint update[NT];
        update[i_local] = 0;
    
        float Lh = 0.5*L;
        float dx2;
        const float delta_x2 = a0 * a0 / D;
    
        for(uint i = i_global; i < D * N; i += NB * NT){
            dx2 = x[i] - positionMemory[i];
            if(dx2 > Lh){
                dx2 -= L;
            }
            if(dx2 < -Lh){
                dx2 += L;
            }
            dx2 *= dx2;
            if(dx2 > delta_x2){
                update[i_local] = 1;
            }
        }
        __syncthreads();
        //only for i_local = 0
        if(!i_local){
            uint foo = 0;
            for(uint i = 0; i < NT; i++){
                foo += update[i];
            }
            atomicAdd(needUpdate, foo);
        }
    }
    void judgeUpdateGrid(Box* box){
    
        checkGrid<<<NB,NT>>>(box->needUpdate_dev, box->L, box->p.x_dev, box->positionMemory);
        uint needUpdate;
        cudaMemcpy(&needUpdate, box->needUpdate_dev, sizeof(uint), cudaMemcpyDeviceToHost);
        if(needUpdate){
            updateGrid2D<<<NB,NT>>>(box, box->grid_dev, box->positionMemory_dev, box->p.x_dev);
        }
    }
}