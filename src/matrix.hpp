#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// TODO: Workaround for missing header if not C++20
#include <bit>

#include <boost/tokenizer.hpp>

#include "types.cuh"
#include "helpers.cuh"

namespace cross {

namespace impl {
    // namespace binary {

    //     template<typename T>
    //     bool try_read_value(std::istream& in, T& out) {
    //         unsigned char *out_bytes = reinterpret_cast<unsigned char*>(&out);
    //         if (!in.read(out_bytes, sizeof(T))) {
    //             if (in.gcount() != 0) {
    //                 // TODO: Invalid file format or error reading
    //                 throw std::runtime_error{"Invalid input file format"};
    //             }
    //             else {
    //                 // TODO: Most likely end of file
    //                 return false;
    //             }
    //         }

    //         if (std::endian::native == std::endian::big) {
    //             std::reverse(out_bytes, out_bytes + sizeof(T));
    //         }
    //         return true;
    //     }

    //     dsize2_t read_size(std::istream& in) {
    //         dsize_t width, height;
    //         if (!try_read_value(in, width)) {
    //             // TODO: Invalid file format or error reading
    //             throw std::runtime_error{"Invalid input file format"};
    //         }

    //         if (!try_read_value(in, height)) {
    //             // TODO: Invalid file format or error reading
    //             throw std::runtime_error{"Invalid input file format"};
    //         }

    //         return dsize2_t{width, height};
    //     }

    //     template <typename T>
    //     std::vector<T> read_matrix_binary(std::istream& in) {
    //         // TODO: Implement binary file loading
    //     }

    // }

    namespace csv {
        dsize2_t read_size(std::istream& in) {
            std::string line;
            if (!std::getline(in, line)) {
                // TODO: Error
                throw std::runtime_error{"Could not read size from file"};
            }
            boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
            auto it = tok.begin();
            if (it == tok.end()) {
                throw std::runtime_error{"Missing matrix width"};
            }
            dsize_t width = std::stoi(*it);

            if (++it == tok.end()) {
                throw std::runtime_error("Missing matrix height");
            }
            dsize_t height = std::stoi(*it);

            if (++it != tok.end()) {
                throw std::runtime_error("More than two dimensions are not supported");
            }
            return dsize2_t{width, height};
        }

        template <typename CONT>
        void read_data(std::istream& in, dsize2_t matrix_size, CONT& out) {
            for (dsize_t y = 0; y < matrix_size.y; ++y) {
                std::string line;
                if (!std::getline(in, line)) {
                    // TODO: Error
                    throw std::runtime_error{std::string{"Failed to read line "} + std::to_string(y) + " from csv file"};
                }

                boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
                auto it = tok.begin();
                dsize_t x = 0;
                for (;it != tok.end() && x < matrix_size.x; ++x, ++it) {
                    out[dsize2_t{x, y}.linear_idx(matrix_size.x)] = parser<typename CONT::value_type>::from_string(*it);
                }

                if (it != tok.end()) {
                    throw std::runtime_error{std::string{"Line "} + std::to_string(y) + " is too long, expected " + std::to_string(matrix_size.x) + "values"};
                }

                if (x != matrix_size.x) {
                    throw std::runtime_error{std::string{"Line"} + std::to_string(y) + " is too short, expected " + std::to_string(matrix_size.x) + ", got " + std::to_string(x) + " values"};
                }
            }
            // TODO: Check that we read the whole file
        }


        void write_size(std::ostream& out, dsize2_t size) {
            out << size.x << "," << size.y << "\n";
        }

        template <typename CONT>
        void write_data(std::ostream& out, dsize2_t matrix_size, const CONT& data) {
            // TODO: Set precision
            out << std::fixed;
            for (dsize_t y = 0; y < matrix_size.y; ++y) {
                auto col_sep = "";
                for (dsize_t x = 0; x < matrix_size.x; ++x) {
                    out << col_sep << data[y * matrix_size.y + x];
                    col_sep = ",";
                }
                out << "\n";
            }
        }

    }
}

template<typename T, typename ALLOC>
class matrix {
public:
    // static matrix<T, ALLOC> load_from_binary(std::istream& in) {

    //     std::vector<T> data(width * height);
    //     for (std::size_t i = 0; i < width * height; ++i) {
    //         if (!impl::try_read_value(in, data[i])) {
    //             // TODO: Invalid file format or error reading
    //             throw std::runtime_error{"Invalid input file format"};
    //         }
    //     }

    //     return matrix{};
    // }

    using value_type = T;

    matrix()
        :matrix(dsize2_t{0, 0})
    {

    }

    explicit matrix(dsize2_t size)
        :size_(size), data_(size.area())
    { }

    static matrix<T, ALLOC> load_from_csv(std::istream& in) {
        auto size = impl::csv::read_size(in);

        std::vector<T, ALLOC> data(size.area());
        impl::csv::read_data(in, size, data);
        return matrix{size, std::move(data)};
    }

    void store_to_csv(std::ostream& out) const {
        impl::csv::write_size(out, size_);
        impl::csv::write_data(out, size_, data_);
    }

    value_type* data() {
        return data_.data();
    }

    const value_type* data() const {
        return data_.data();
    }

    dsize2_t size() const {
        return size_;
    }

    dsize_t area() const {
        return size_.area();
    }
private:
    dsize2_t size_;

    std::vector<T, ALLOC> data_;

    matrix(dsize2_t size, std::vector<T, ALLOC>&& data)
        :size_(size), data_(std::move(data))
    { }
};

template<typename T, typename ALLOC>
std::ostream& operator<<(std::ostream& out, const matrix<T, ALLOC>& matrix) {
    matrix.store_to_csv(out);
    return out;
}

}