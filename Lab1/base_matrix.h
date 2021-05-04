#pragma once

template<typename DerrivedMatrix>
class BaseMatrix
{
public:
    BaseMatrix(size_t height = 0, size_t width = 0, double *data = nullptr) :
        _width(width), _height(height), _data(data)
    {
    }

    BaseMatrix(const BaseMatrix &other) : BaseMatrix(other._height, other._width, other._data)
    {
    }

    BaseMatrix &operator=(const BaseMatrix &other)
    {
        _height = other._height;
        _width = other._width;
        _data = other._data;

        return *this;
    }

    BaseMatrix(BaseMatrix &&other) : BaseMatrix(other)
    {
        other._data = nullptr;
    }

    virtual ~BaseMatrix()
    {
    }

    virtual DerrivedMatrix multiply(const DerrivedMatrix &other) const = 0;
    virtual double at(size_t i, size_t j) const = 0;

    inline size_t index(size_t i, size_t j) const
    {
        return i * _width + j;
    }

    inline size_t height() const
    {
        return _height;
    }

    inline size_t width() const
    {
        return _width;
    }

    inline double *data() const
    {
        return _data;
    }

    inline size_t dataSize() const
    {
        return sizeof(double) * width() * height();
    }

protected:
    size_t _width;
    size_t _height;
    double *_data;
};