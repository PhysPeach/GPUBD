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

    //setters and getters
    __global__ void setRndPositions(float l,float *diam_dev, float *x_dev, float *v_dev ,curandState *rndState_dev){
        unsigned int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        
        float atmp = a2 - a1;
        for(unsigned int i = i_global; i < N; i+=NB*NT){
            diam_dev[i] = a1 + atmp * (i%2);
        }

        curandState localState = rndState_dev[i_global];
        for(unsigned int i = i_global; i < D * N; i+=NB*NT){
            x_dev[i] = l * curand_uniform(&localState);
            v_dev[i] = 0.0;
        }
        rndState_dev[i_global] = localState;
    }

    void initParticles(Particles* p, float L){
        //avoiding super overraps
        float Ltmp = L - 0.5 * (a1+a2);

        //set positions by uniform random destribution
        setRndPositions<<<NB,NT>>>(Ltmp, p->diam_dev, p->x_dev, p->v_dev, p->rndState_dev);
    }
}