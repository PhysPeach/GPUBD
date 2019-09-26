#include "../h/particles.cuh"

namespace PhysPeach{
    __global__ void init_genrand_kernel(unsigned long long seed, curandState* state){
        unsigned int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, i_global,0,&state[i_global]);
    }
    
    void makeParticles(Particles* p){
        //malloc host
        p->diam = (float*)malloc(N*sizeof(float));
        p->x = (float*)malloc(D*N*sizeof(float));
        p->v = (float*)malloc(D*N*sizeof(float));
    
        //malloc device
        cudaMalloc((void**)&p->diam_dev, N*sizeof(float));
        cudaMalloc((void**)&p->x_dev, D*N*sizeof(float));
        cudaMalloc((void**)&p->v_dev, D*N*sizeof(float));
        cudaMalloc((void**)&p->rndState_dev, D*N*sizeof(curandState));
        cudaMalloc((void**)&p->force_dev, D*N*sizeof(float));
    
        //set rnd seed
        init_genrand_kernel<<<NB,NT>>>((unsigned long long)genrand_int32(),p->rndState_dev);
        return;
    }
    
    void killParticles(Particles* p){
        free(p->diam);
        free(p->x);
        free(p->v);
        
        cudaFree(p->diam_dev);
        cudaFree(p->x_dev);
        cudaFree(p->v_dev);
        cudaFree(p->rndState_dev);
        cudaFree(p->force_dev);
        return;
    }
}