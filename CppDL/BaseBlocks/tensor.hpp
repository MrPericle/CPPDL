#ifndef CPPDL_BASEBLOCKS_LOSSHPP
#define CPPDL_BASEBLOCKS_LOSSHPP

#include <iostream>
#include <vector>
#include <numeric>   // for std::accumulate
#include <stdexcept> // for std::runtime_error
#include <cmath>     // for std::sqrt

namespace dl {

    // Forward declaration for recursive template
    template <std::size_t N>
    class Tensor;

    // Specialization for one-dimensional Tensor (N=1)
    template <>
    class Tensor<1> : public std::vector<double> {
    public:
        Tensor() : std::vector<double>() {}
        Tensor(size_t n, double value = 0.0) : std::vector<double>(n, value) {}

        // Custom methods
        double sum() const {
            return std::accumulate(this->begin(), this->end(), 0.0);
        }

        double average() const {
            if (this->empty()) {
                throw std::runtime_error("Cannot compute average of an empty vector.");
            }
            return sum() / this->size();
        }

        void print() const {
            for (const auto& value : *this) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }

        // Element-wise operations
        Tensor operator+(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Vectors must be of the same size for element-wise addition.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] + other[i];
            }
            return result;
        }

        Tensor operator-(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Vectors must be of the same size for element-wise subtraction.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] - other[i];
            }
            return result;
        }

        Tensor operator*(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Vectors must be of the same size for element-wise multiplication.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] * other[i];
            }
            return result;
        }

        Tensor operator/(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Vectors must be of the same size for element-wise division.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                if (other[i] == 0.0) {
                    throw std::invalid_argument("Division by zero encountered in element-wise division.");
                }
                result[i] = (*this)[i] / other[i];
            }
            return result;
        }

        // Scalar operations
        Tensor operator+(double scalar) const {
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] + scalar;
            }
            return result;
        }

        Tensor operator-(double scalar) const {
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] - scalar;
            }
            return result;
        }

        Tensor operator*(double scalar) const {
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] * scalar;
            }
            return result;
        }

        Tensor operator/(double scalar) const {
            if (scalar == 0.0) {
                throw std::invalid_argument("Division by zero encountered.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] / scalar;
            }
            return result;
        }
    };

    // Recursive case: Multi-dimensional tensor specialized for double
    template <std::size_t N>
    class Tensor : public std::vector<Tensor<N-1>> {
    public:
        // Constructors
        Tensor() : std::vector<Tensor<N-1>>() {}
        Tensor(size_t n) : std::vector<Tensor<N-1>>(n) {}

        // Custom methods for higher-dimensional tensors
        double sum() const {
            double total = 0.0;
            for (const auto& tensor : *this) {
                total += tensor.sum();
            }
            return total;
        }

        double mean() const {
            size_t total_elements = 0;
            for (const auto& tensor : *this) {
                total_elements += tensor.size();
            }
            if (total_elements == 0) {
                throw std::runtime_error("Cannot compute mean of an empty tensor.");
            }
            return sum() / total_elements;
        }

        double stdDev() const {
            if (this->size() <= 1) {
                throw std::runtime_error("Cannot compute standard deviation with fewer than 2 elements.");
            }
            double mean_val = mean();
            double variance = 0.0;
            for (const auto& tensor : *this) {
                variance += (tensor.mean() - mean_val) * (tensor.mean() - mean_val);
            }
            variance /= this->size() - 1;
            return std::sqrt(variance);
        }

        void print() const {
            for (const auto& tensor : *this) {
                tensor.print();
            }
        }

        // Element-wise operations
        Tensor operator+(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Tensors must be of the same size for element-wise addition.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] + other[i];
            }
            return result;
        }

        Tensor operator-(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Tensors must be of the same size for element-wise subtraction.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] - other[i];
            }
            return result;
        }

        Tensor operator*(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Tensors must be of the same size for element-wise multiplication.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] * other[i];
            }
            return result;
        }

        Tensor operator/(const Tensor& other) const {
            if (this->size() != other.size()) {
                throw std::invalid_argument("Tensors must be of the same size for element-wise division.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] / other[i];
            }
            return result;
        }

        // Scalar operations
        Tensor operator+(double scalar) const {
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] + scalar;
            }
            return result;
        }

        Tensor operator-(double scalar) const {
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] - scalar;
            }
            return result;
        }

        Tensor operator*(double scalar) const {
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] * scalar;
            }
            return result;
        }

        Tensor operator/(double scalar) const {
            if (scalar == 0.0) {
                throw std::invalid_argument("Division by zero encountered.");
            }
            Tensor result(this->size());
            for (size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] / scalar;
            }
            return result;
        }
    };
};

#endif
