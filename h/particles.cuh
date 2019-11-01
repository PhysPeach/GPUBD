#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include <time.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "../h/MT.h"
#include "../h/generalFuncs.cuh"
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
        float *getNK_dev[2]; //K[D * N]
        float* Nvg_dev[D][2];
    };
    void makeParticles(Particles* particles);
    void killParticles(Particles* particles);
    void scatterParticles(Particles* particles, float L);

    //time evolutions
    __global__ void vEvoBD(float *v, double dt, float themalFuctor, float *force, curandState *state);
    __global__ void xEvo(float *x, double dt, float L, float *v);
    void removevg2D(Particles* p);

    //setters and getters

    float K(Particles* particles);
}

#endif