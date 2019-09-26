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
    //for grid
}