#include "../h/generalFuncs.cuh"

namespace PhysPeach{
    __global__ void setIntVecZero(uint* x, uint Num){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(uint i = i_global; i < Num; i += NB * NT){
            x[Num] = 0;
        }
    }
}