#pragma once

#include "base_matrix.h"
#include "cpu_matrix.h"

struct CudaMatrixData
{
    size_t height;
    size_t width;
    double *data;
};

class GPUMatrix : public BaseMatrix<GPUMatrix>
{
public:
    GPUMatrix(const CPUMatrix &m);
    GPUMatrix(const GPUMatrix &other);
    GPUMatrix(GPUMatrix &&other);
    GPUMatrix &operator=(const GPUMatrix &other);

    ~GPUMatrix();

    CPUMatrix toCPU();

    double at(size_t i, size_t j) const override;

    GPUMatrix multiply(const GPUMatrix &other) const override;
    CPUMatrix multiply(const CPUMatrix &other) const;

protected:
    CudaMatrixData *_cudaData;

    GPUMatrix(size_t height = 0, size_t width = 0);
};

