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
    };
    void makeParticles(Particles* particles);
    void killParticles(Particles* particles);
    void initParticles(Particles* particles, float L);

    //inline float getNK(Particles* particles);
    
    //inline void setvgzero2D(Particles* particles);

    //__global__ void vDvlpBD(Particles* particles, float *v_dev, Box* box, float *force_dev, curandState *rndState_dev);
    //__global__ void xDvlp(Particles* float *x_dev, Box* box, float *v_dev);

}

#endif