#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include <cuda.h>
#include <curand_kernel.h>

#include "../h/MT.h"

#include "../h/parameters.cuh"

namespace PhysPeach{
    struct Particles{
        float *diam;
        float *x;
        float *v;

        float *diam_dev;
        float *x_dev;
        float *v_dev;
        curandState *rndState_dev;
        float *force_dev;

        //for setters and getters
        float *getNK_dev[2], *getNU_dev[2]; //K[D * N], U[N]
        float* getNvg_dev;
        float* Nvg_dev;
    };
    void makeParticles(Particles* particles);
    void killParticles(Particles* particles);
    void scatterParticles(Particles* particles, float L);
    __global__ void checkPeriodic(float L, float *x);

    //inline float getNK(Particles* particles);
    
    //inline void setvgzero2D(Particles* particles);

    //__global__ void vDvlpBD(Particles* particles, float *v_dev, Box* box, float *force_dev, curandState *rndState_dev);
    //__global__ void xDvlp(Particles* float *x_dev, Box* box, float *v_dev);

}

#endif