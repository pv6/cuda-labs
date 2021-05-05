#include <random>
#include <iostream>
#include <windows.h>

#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include "gpu_sh_mem_matrix.h"

template <typename Matrix>
struct TestResult
{
    double time;
    Matrix resultMatrix;
};

template <typename Matrix>
static TestResult<Matrix> testPerformance(const Matrix &a, const Matrix &b)
{
    LARGE_INTEGER startTime, endTime, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&startTime);

    Matrix resultMatrix = a.multiply(b);

    QueryPerformanceCounter(&endTime);

    double time = (double)(endTime.QuadPart - startTime.QuadPart) / freq.QuadPart;

    return { time, std::move(resultMatrix) };
}

static CPUMatrix generateRandomMatrix(size_t height, size_t width)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(-3, 3);

    CPUMatrix output(height, width);

    for (size_t i = 0; i < height; i++)
        for (size_t j = 0; j < width; j++)
            output.at(i, j) = distrib(gen);

    return output;
}

int main(int argc, char *argv[])
{
    const size_t MATRIX_SIZE[][2] = {
        {1000, 1100},
        {1500, 1600},
        {2000, 2100}
    };
    const size_t N = ARRAYSIZE(MATRIX_SIZE);

    for (size_t i = 0; i < N; i++)
    {
        std::cout << "--------------" << std::endl;
        std::cout << "Matrix A size: " << MATRIX_SIZE[i][0] << "x" << MATRIX_SIZE[i][1] << std::endl;
        std::cout << "Matrix B size: " << MATRIX_SIZE[i][1] << "x" << MATRIX_SIZE[i][0] << std::endl;

        CPUMatrix a = generateRandomMatrix(MATRIX_SIZE[i][0], MATRIX_SIZE[i][1]);
        CPUMatrix b = generateRandomMatrix(MATRIX_SIZE[i][1], MATRIX_SIZE[i][0]);

        TestResult<CPUMatrix> cpuResult = testPerformance(a, b);
        TestResult<GPUMatrix> gpuResult = testPerformance(GPUMatrix(a), GPUMatrix(b));
        TestResult<GPUMatrix> gpuShMemResult = testPerformance<GPUMatrix>(GPUShMemMatrix(a), GPUShMemMatrix(b));
        double distance = CPUMatrix::distance(cpuResult.resultMatrix,
            gpuShMemResult.resultMatrix.toCPU());

        std::cout << "Time on CPU: " << cpuResult.time << " s" << std::endl;
        std::cout << "Time on GPU: " << gpuResult.time << " s" << std::endl;
        std::cout << "Time on GPU w/ shared memory: " << gpuShMemResult.time << " s" << std::endl;
        std::cout << "CPU and GPU result distance: " << distance << std::endl;
    }

    return 0;
}
