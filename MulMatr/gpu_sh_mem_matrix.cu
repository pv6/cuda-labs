#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpu_sh_mem_matrix.h"

#define BLOCK_SIZE 32

__global__ void mulMatrShMemKernel(const CudaMatrixData *aPtr, const CudaMatrixData *bPtr,
    CudaMatrixData *res)
{
    const CudaMatrixData a = *aPtr;
    const CudaMatrixData b = *bPtr;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double subB[BLOCK_SIZE][BLOCK_SIZE];

    double resValue = 0;
    for (size_t subIndx = 0; subIndx * BLOCK_SIZE < a.width; subIndx++)
    {
        subA[threadIdx.y][threadIdx.x] = 0;
        subB[threadIdx.y][threadIdx.x] = 0;

        size_t const subX = subIndx * BLOCK_SIZE + threadIdx.x;
        if (subX < a.width && y < a.height)
            subA[threadIdx.y][threadIdx.x] = a.data[y * a.width + subX];

        size_t const subY = subIndx * BLOCK_SIZE + threadIdx.y;
        if (x < b.width && subY < b.height)
            subB[threadIdx.y][threadIdx.x] = b.data[subY * b.width + x];

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; k++)
            resValue += subA[threadIdx.y][k] * subB[k][threadIdx.x];

        __syncthreads();
    }

    if (y < res->height && x < res->width)
        res->data[y * res->width + x] = resValue;
}

void GPUShMemMatrix::callKernel(const dim3 &cudaBlocks, const dim3 &cudaThreads,
    const CudaMatrixData *a, const CudaMatrixData *b, CudaMatrixData *result) const
{
    mulMatrShMemKernel<<<cudaBlocks, cudaThreads>>>(a, b, result);
}
