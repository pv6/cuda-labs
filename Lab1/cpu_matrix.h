#pragma once

#include "base_matrix.h"

class CPUMatrix : public BaseMatrix<CPUMatrix>
{
public:
    CPUMatrix(size_t height = 0, size_t width = 0);
    CPUMatrix(const CPUMatrix &other);
    CPUMatrix(CPUMatrix &&other);
    CPUMatrix &operator=(const CPUMatrix &other);

    ~CPUMatrix();

    double &at(size_t i, size_t j);
    double at(size_t i, size_t j) const override;

    CPUMatrix multiply(const CPUMatrix &other) const override;

    static double distance(const CPUMatrix &a, const CPUMatrix &b);
};