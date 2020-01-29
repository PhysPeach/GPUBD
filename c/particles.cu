#include "../h/particles.cuh"

namespace PhysPeach{
    __global__ void init_genrand_kernel(unsigned long long s, curandState* state){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(s, i_global,0,&state[i_global]);
    }
    
    void makeParticles(Particles* p){
        //malloc host
        p->diam = (float*)malloc(NP * sizeof(float));
        p->x = (double*)malloc(D * NP * sizeof(double));
        p->v = (float*)malloc(D * NP * sizeof(float));
    
        //malloc device
        cudaMalloc((void**)&p->diam_dev, NP * sizeof(float));
        cudaMalloc((void**)&p->x_dev, D * NP * sizeof(double));
        cudaMalloc((void**)&p->v_dev, D * NP * sizeof(float));
        cudaMalloc((void**)&p->rndState_dev, D * NP *sizeof(curandState));
        cudaMalloc((void**)&p->force_dev, D * NP*sizeof(float));
        
        //for setters and getters 
        cudaMalloc((void**)&p->getNK_dev[0], D * NP * sizeof(float));
        cudaMalloc((void**)&p->getNK_dev[1], D * NP * sizeof(float));
        for(uint i = 0; i < D; i++){
            cudaMalloc((void**)&p->Nvg_dev[i][0], NP * sizeof(float));
            cudaMalloc((void**)&p->Nvg_dev[i][1], NP * sizeof(float));
        }
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

        cudaFree(p->getNK_dev[0]);
        cudaFree(p->getNK_dev[1]);
        for(uint i = 0; i< D; i++){
            cudaFree(p->Nvg_dev[i][0]);
            cudaFree(p->Nvg_dev[i][1]);
        }
        return;
    }

    //setters and getters
    __global__ void setRndParticleStates(double l,float *diam, double *x, float *v ,curandState *rndState){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        
        float atmp = a2 - a1;
        for(uint i = i_global; i < NP; i+=NB*NT){
            diam[i] = a1 + atmp * (i%2);
        }
        for(uint i = i_global; i < D * NP; i+=NB*NT){
            x[i] = l * curand_uniform(&rndState[i]);
            v[i] = 0.0;
        }
    }
    void scatterParticles(Particles* p, double L){
        //set positions by uniform random destribution
        init_genrand((unsigned long)time(NULL));
        init_genrand_kernel<<<NB,NT>>>((unsigned long long)genrand_int32(),p->rndState_dev);
        setRndParticleStates<<<NB,NT>>>(0.9999*L, p->diam_dev, p->x_dev, p->v_dev, p->rndState_dev);
        return;
    }

    //time evolutions
    __global__ void vEvoLD(float *v, double dt, float thermalFuctor, float *force, curandState *state){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = i_global; i < D * NP; i += NB*NT){
            v[i] += dt*(-v[i] + force[i] + thermalFuctor*curand_normal(&state[i]));
        }
    }
    __global__ void xEvoLD(double *x, double dt, double L, float *v){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = i_global; i < D * NP; i += NB*NT){
            x[i] += dt * v[i];
            //periodic
            if(x[i] > L){
                x[i] -= L;
            }
            else if(x[i] < 0){
                x[i] += L;
            }
        }
    }
    __global__ void halfvEvoMD(float *v, double dt, float *force){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = i_global; i < D * NP; i += NB*NT){
            v[i] += 0.5*dt*force[i];
        }
    }
    __global__ void xEvoMD(double *x, double dt, double L, float *v, float *force){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = i_global; i < D * NP; i += NB*NT){
            x[i] += dt * v[i] + 0.5*dt*dt*force[i];
            //periodic
            if(x[i] > L){
                x[i] -= L;
            }
            else if(x[i] < 0){
                x[i] += L;
            }
        }
    }
    __global__ void glo_removevg(float *v, float* Nvg){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;

        float vg_local = Nvg[0]/NP;
        for(uint i = i_global; i < NP; i += NB * NT){
            v[i] -= vg_local;
        }
    }
    void removevg2D(Particles* p){
        //summations
        uint flip = 0;
        uint l = NP;
        reductionSum<<<NB,NT>>>(p->Nvg_dev[0][0], &p->v_dev[0], l);
        reductionSum<<<NB,NT>>>(p->Nvg_dev[1][0], &p->v_dev[NP], l);
        l = (l + NT-1)/NT;
        while(l > 1){
            flip = !flip;
            reductionSum<<<NB,NT>>>(p->Nvg_dev[0][flip], p->Nvg_dev[0][!flip], l);
            reductionSum<<<NB,NT>>>(p->Nvg_dev[1][flip], p->Nvg_dev[1][!flip], l);
            l = (l + NT-1)/NT;
        }
        glo_removevg<<<NB,NT>>>(&p->v_dev[0],p->Nvg_dev[0][flip]);
        glo_removevg<<<NB,NT>>>(&p->v_dev[NP],p->Nvg_dev[1][flip]);
        
        return;
    }
    __global__ void getvv(float *vv, float *v){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(uint i = i_global; i < D*NP; i += NB * NT){
            vv[i] = v[i] * v[i];
        }
    }
    float K(Particles* p){
        float NK = 0;
        uint flip = 0;

        getvv<<<NB,NT>>>(p->getNK_dev[0], p->v_dev);
        //summations
        for(uint l = D * NP; l > 1; l = (l + NT-1)/NT){
            flip = !flip;
            reductionSum<<<NB,NT>>>(p->getNK_dev[flip], p->getNK_dev[!flip], l);
        }
        cudaMemcpy(&NK, p->getNK_dev[flip], sizeof(float), cudaMemcpyDeviceToHost);

        return NK/(2. * NP);
    }
}