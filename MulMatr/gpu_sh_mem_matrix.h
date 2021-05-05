#pragma once

#include "gpu_matrix.h"

class GPUShMemMatrix : public GPUMatrix
{
public:
    using GPUMatrix::GPUMatrix;

    void callKernel(const dim3 &cudaBlocks, const dim3 &cudaThreads,
        const CudaMatrixData *a, const CudaMatrixData *b, CudaMatrixData *result) const override;
};
