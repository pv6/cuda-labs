#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <stdexcept>

#include "gpu_matrix.h"

void GPUMatrix::allocateMemory()
{
    cudaMalloc(&_data, dataSize());

    cudaMalloc(&_cudaData, sizeof(CudaMatrixData));
    CudaMatrixData data({ _height, _width, _data });
    cudaMemcpy(_cudaData, &data, sizeof(CudaMatrixData), cudaMemcpyHostToDevice);
}

void GPUMatrix::freeMemory()
{
    cudaFree(_data);
    cudaFree(_cudaData);
}

GPUMatrix::GPUMatrix(size_t height, size_t width) : BaseMatrix(height, width)
{
    allocateMemory();
}

GPUMatrix::GPUMatrix(const CPUMatrix &m) : GPUMatrix(m.height(), m.width())
{
    cudaMemcpy(_data, m.data(), dataSize(), cudaMemcpyHostToDevice);
}

GPUMatrix::GPUMatrix(const GPUMatrix &other) : GPUMatrix(other.height(), other.width())
{
    cudaMemcpy(_data, other.data(), dataSize(), cudaMemcpyDeviceToDevice);
}

GPUMatrix::GPUMatrix(GPUMatrix &&other) : BaseMatrix(std::move(other))
{
    _cudaData = other._cudaData;
    other._cudaData = nullptr;
}

GPUMatrix &GPUMatrix::operator=(const GPUMatrix &other)
{
    freeMemory();

    _height = other._height;
    _width = other._width;
    
    allocateMemory();

    cudaMemcpy(_data, other.data(), dataSize(), cudaMemcpyDeviceToDevice);

    return *this;
}

GPUMatrix::~GPUMatrix()
{
    freeMemory();
}

CPUMatrix GPUMatrix::toCPU()
{
    CPUMatrix output(height(), width());
    cudaMemcpy(output.data(), _data, dataSize(), cudaMemcpyDeviceToHost);
    return output;
}

double GPUMatrix::at(size_t i, size_t j) const
{
    double x;
    cudaMemcpy(&x, _data + index(i, j), sizeof(double), cudaMemcpyDeviceToHost);
    return x;
}

__global__ void mulMatrKernel(const CudaMatrixData *a, const CudaMatrixData *b,
    CudaMatrixData *res)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= res->height || j >= res->width)
        return;

    size_t index = i * res->width + j;
    res->data[index] = 0;
    for (size_t k = 0; k < a->width; k++)
        res->data[index] += a->data[i * a->width + k] * b->data[k * b->width + j];
}

void GPUMatrix::callKernel(const dim3 &cudaBlocks, const dim3 &cudaThreads,
    const CudaMatrixData *a, const CudaMatrixData *b, CudaMatrixData *result) const
{
    mulMatrKernel<<<cudaBlocks, cudaThreads>>>(a, b, result);
}

GPUMatrix GPUMatrix::multiply(const GPUMatrix &other) const
{
    if (_width != other._height)
        throw std::runtime_error("Wrong matrtix dimensions: a.width != b.height");

    GPUMatrix result = GPUMatrix(_height, other._width);

    static const unsigned int BLOCK_SIZE = 32;
    dim3 cudaThreads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 cudaBlocks(
        static_cast<unsigned int>(result._width + cudaThreads.x - 1) / cudaThreads.x,
        static_cast<unsigned int>(result._height + cudaThreads.y - 1) / cudaThreads.y
    );

    cudaStream_t stream = 0;
    cudaEvent_t endEvent;
    cudaEventCreate(&endEvent);

    callKernel(cudaBlocks, cudaThreads, _cudaData, other._cudaData, result._cudaData);

    cudaEventRecord(endEvent, stream);
    cudaEventSynchronize(endEvent);

    cudaEventDestroy(endEvent);

    return result;
}

CPUMatrix GPUMatrix::multiply(const CPUMatrix &other) const
{
    return multiply(GPUMatrix(other)).toCPU();
}
