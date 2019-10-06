#include "../h/generalFuncs.cuh"

namespace PhysPeach{
    __global__ void setIntVecZero(uint* x, uint Num){
        uint i_global = blockIdx.x * blockDim.x + threadIdx.x;
        for(uint i = i_global; i < Num; i += NB * NT){
            x[Num] = 0;
        }
    }
    __global__ void reductionMax(float *out, float *in, uint l){
        uint i_block = blockIdx.x;
        uint i_local = threadIdx.x;
        uint i_global = i_block * blockDim.x + i_local;
    
        __shared__ float f[NT];
    
        uint remain, reduce;
        uint ib = i_block;
        for(uint i = i_global; i < l; i += NB*NT){
            f[i_local] = in[i];
            __syncthreads();
    
            for(uint j = NT; j > 1; j = remain){
                reduce = j >> 1;
                remain = j - reduce;
                if((i_local < reduce) && (i + remain < l)){
                    if(f[i_local] < f[i_local + remain]){
                        f[i_local] = f[i_local+remain];
                    }
                }
                __syncthreads();
            }
            if(i_local == 0){
                out[ib] = f[0];
            }
            __syncthreads();
            ib += NB;
        }
    }
}