#include <cstring>
#include <utility>
#include <stdexcept>
#include <future>

#include "cpu_matrix.h"

CPUMatrix::CPUMatrix(size_t height, size_t width) :
    BaseMatrix(height, width, new double[height * width])
{
}

CPUMatrix::CPUMatrix(const CPUMatrix &other) : CPUMatrix(other.height(), other.width())
{
    memcpy(_data, other._data, dataSize());
}

CPUMatrix::CPUMatrix(CPUMatrix &&other) : BaseMatrix(std::move(other))
{
}

CPUMatrix &CPUMatrix::operator=(const CPUMatrix &other)
{
    delete[] _data;

    _height = other._height;
    _width = other._width;
    _data = new double[_height * _width];
    memcpy(_data, other._data, dataSize());

    return *this;
}

CPUMatrix::~CPUMatrix()
{
    delete[] _data;
}

double &CPUMatrix::at(size_t i, size_t j)
{
    return _data[index(i, j)];
}

double CPUMatrix::at(size_t i, size_t j) const
{
    return _data[index(i, j)];
}

CPUMatrix CPUMatrix::multiply(const CPUMatrix &other) const
{
    if (width() != other.height())
        throw std::runtime_error("Wrong matrtix dimensions: a.width != b.width");

    CPUMatrix result(height(), other.width());

    const size_t N = std::max(1U, std::thread::hardware_concurrency());
    const size_t ROWS_PER_JOB = (size_t)ceil((double)result.height() / N);

    auto multiplyRows = [N, ROWS_PER_JOB](const CPUMatrix *a, const CPUMatrix *b,
        CPUMatrix *result, size_t index)
    {
        const size_t
            startRow = std::min(index * ROWS_PER_JOB, result->height()),
            endRow = std::min((index + 1) * ROWS_PER_JOB, result->height());

        for (size_t i = startRow; i < endRow; i++)
            for (size_t j = 0; j < result->width(); j++)
            {
                double &x = result->at(i, j);

                x = 0;
                for (size_t k = 0; k < a->width(); k++)
                    x += a->at(i, k) * b->at(k, j);
            }
    };

    std::vector<std::future<void>> jobs(N);
    for (int i = 0; i < N; i++)
        jobs[i] = std::async(std::launch::async, multiplyRows, this, &other, &result, i);
    for (const auto &job : jobs)
        job.wait();

    //for (size_t i = 0; i < result.height(); i++)
    //    for (size_t j = 0; j < result.width(); j++)
    //    {
    //        double &x = result.at(i, j);
    //
    //        x = 0;
    //        for (size_t k = 0; k < width(); k++)
    //            x += at(i, k) * other.at(k, j);
    //    }

    return result;
}

double CPUMatrix::distance(const CPUMatrix &a, const CPUMatrix &b)
{
    if (a.width() != b.width() || a.height() != b.height())
        throw std::runtime_error("Matrix dimensions are not equal!");

    const size_t N = a.width() * a.height();
    double res = 0.0;
    for (size_t i = 0; i < N; i++)
        res += std::abs(a._data[i] - b._data[i]);

    return res;
}