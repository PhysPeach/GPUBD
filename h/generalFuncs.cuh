#ifndef GENERAL_CUH
#define GENERAL_CUH

#include <math.h>

#include <cuda.h>
#include "parameters.cuh"

namespace PhysPeach{
    __global__ void setIntVecZero(uint* x, uint Num);
    __global__ void reductionMax(float *out, float *in, uint l);


}

#endif