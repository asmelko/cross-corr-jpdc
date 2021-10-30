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

class no_padding {
public:
    static dsize2_t total_size(dsize2_t matrix_size) {
        return matrix_size;
    }

    template<typename IT>
    static void pad_row(IT begin, IT end) {
        return;
    }
};

template<dsize_t MULT>
class relative_zero_padding {
public:
    static dsize2_t total_size(dsize2_t matrix_size) {
        return matrix_size * MULT;
    }

    template<typename IT>
    static void pad_row(IT begin, IT end) {
        for (auto it = begin; it != end; ++it) {
            *it = 0;
        }
    }
};

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

    //     dsize2_t read_matrix_size(std::istream& in) {
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
        inline dsize_t read_num_matrices(std::istream& in) {
            std::string line;
            if (!std::getline(in, line)) {
                // TODO: Error
                throw std::runtime_error{"Could not read number of matricies from file"};
            }
            boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
            auto it = tok.begin();
            if (it == tok.end()) {
                throw std::runtime_error{"Missing number of matricies width"};
            }
            dsize_t num = std::stoi(*it);

            if (++it != tok.end()) {
                throw std::runtime_error("Unexpected value, expected just one number");
            }
            return num;
        }

        inline dsize2_t read_matrix_size(std::istream& in) {
            std::string line;
            if (!std::getline(in, line)) {
                // TODO: Error
                throw std::runtime_error{"Could not read matrix size from file"};
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

        template <typename T, typename PADDING>
        void read_data(std::istream& in, dsize2_t matrix_size, dsize2_t padded_size, T* out) {
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
                    out[dsize2_t{x, y}.linear_idx(padded_size.x)] = from_string<T>(*it);
                }

                if (it != tok.end()) {
                    throw std::runtime_error{std::string{"Line "} + std::to_string(y) + " is too long, expected " + std::to_string(matrix_size.x) + "values"};
                }

                if (x != matrix_size.x) {
                    throw std::runtime_error{std::string{"Line"} + std::to_string(y) + " is too short, expected " + std::to_string(matrix_size.x) + ", got " + std::to_string(x) + " values"};
                }

                dsize_t padding_start_idx = dsize2_t{x, y}.linear_idx(padded_size.x);
                dsize_t padding_end_idx = padding_start_idx + padded_size.x - x;
                PADDING::pad_row(
                    out + padding_start_idx,
                    out + padding_end_idx
                );
            }
            // TODO: Check that we read the whole file

            for (dsize_t y = matrix_size.y; y < padded_size.y; ++y) {
                PADDING::pad_row(
                    out + dsize2_t{0, y}.linear_idx(padded_size.x),
                    out + dsize2_t{padded_size.x, y}.linear_idx(padded_size.x)
                );
            }
        }

        inline void write_num_matrices(std::ostream& out, dsize_t num_matrices) {
            out << num_matrices << "\n";
        }

        inline void write_matrix_size(std::ostream& out, dsize2_t size) {
            out << size.x << "," << size.y << "\n";
        }

        template <typename T>
        void write_data(std::ostream& out, dsize2_t matrix_size, const T* data) {
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

template<typename T>
class matrix_view {
public:
    using value_type = T;
    using size_type = dsize2_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using iterator = pointer;
    using const_iterator = const_pointer;

    matrix_view(dsize2_t size, pointer data)
        :size_(size), data_(data)
    { }

    pointer data() {
        return data_;
    }

    const_pointer data() const {
        return data_;
    }

    dsize2_t size() const {
        return size_;
    }

    dsize_t area() const {
        return size_.area();
    }

    const_iterator begin() const {
        return data();
    }

    iterator begin() {
        return data();
    }

    const_iterator end() const {
        return data() + area();
    }

    iterator end() {
        return data() + area();
    }

    reference operator[](size_type i) {
        return data_[i.linear_idx(size_.x)];
    }

    const_reference operator[](size_type i) const {
        return data_[i.linear_idx(size_.x)];
    }

private:
    dsize2_t size_;

    pointer data_;
};

template<typename T>
class const_matrix_view {
public:
    using value_type = T;
    using size_type = dsize2_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using iterator = pointer;
    using const_iterator = const_pointer;

    const_matrix_view(dsize2_t size, const_pointer data)
        :size_(size), data_(data)
    { }

    const_pointer data() const {
        return data_;
    }

    dsize2_t size() const {
        return size_;
    }

    dsize_t area() const {
        return size_.area();
    }

    const_iterator begin() const {
        return data();
    }

    const_iterator end() const {
        return data() + area();
    }

    const_reference operator[](size_type i) const {
        return data_[i.linear_idx(size_.x)];
    }

private:
    dsize2_t size_;

    const_pointer data_;
};

template<typename T, typename ALLOC = std::allocator<T>>
class data_single {
public:
    using value_type = T;
    using allocator_type = ALLOC;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<ALLOC>::pointer;
    using const_pointer = typename std::allocator_traits<ALLOC>::const_pointer;

    using iterator = pointer;
    using const_iterator = const_pointer;

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

    data_single()
        :data_single<T, ALLOC>(dsize2_t{0,0})
    { }

    data_single(dsize2_t size)
        :size_(size), data_(size.area())
    { }

    template<typename PADDING>
    static data_single<T, ALLOC> load_from_csv(std::istream& in) {
        auto size = impl::csv::read_matrix_size(in);
        auto padded_size = PADDING::total_size(size);

        std::vector<T, ALLOC> data(padded_size.area());
        impl::csv::read_data<T, PADDING>(in, size, padded_size, data.data());

        return data_single{padded_size, std::move(data)};
    }

    void store_to_csv(std::ostream& out) const {
        impl::csv::write_matrix_size(out, size_);
        impl::csv::write_data(out, size_, data_.data());
    }

    size_type size() const {
        return size_.area();
    }

    dsize2_t matrix_size() const {
        return size_;
    }

    pointer data() {
        return data_.data();
    }

    const_pointer data() const {
        return data_.data();
    }

    iterator begin() {
        return data();
    }

    const_iterator begin() const {
        return data();
    }

    iterator end() {
        return data() + size();
    }

    const_iterator end() const {
        return data() + size();
    }

    matrix_view<T> view() {
        return matrix_view<T>{
            size_,
            data_.data()
        };
    }

    const_matrix_view<T> view() const {
        return const_matrix_view<T>{
            size_,
            data_.data()
        };
    }
private:
    dsize2_t size_;

    std::vector<T, ALLOC> data_;

    data_single(dsize2_t size, std::vector<T, ALLOC>&& data)
        :size_(size), data_(std::move(data))
    { }
};

template<typename T, typename ALLOC>
std::ostream& operator<<(std::ostream& out, const data_single<T, ALLOC>& matrix) {
    matrix.store_to_csv(out);
    return out;
}

template<typename T, typename ALLOC = std::allocator<T>>
class data_array {
public:
    using value_type = T;
    using allocator_type = ALLOC;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<ALLOC>::pointer;
    using const_pointer = typename std::allocator_traits<ALLOC>::const_pointer;

    using iterator = pointer;
    using const_iterator = const_pointer;

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

    data_array()
        :data_array<T, ALLOC>(0, dsize2_t{0,0})
    { }

    data_array(dsize_t num_matrices, dsize2_t matrix_size)
        :num_matrices_(num_matrices), matrix_size_(matrix_size), data_(matrix_size.area() * num_matrices)
    { }

    template<typename PADDING>
    static data_array<T, ALLOC> load_from_csv(std::vector<std::ifstream> in) {
        std::vector<dsize2_t> sizes;
        std::vector<dsize2_t> padded_sizes;
        dsize_t total_data_size = 0;
        for (auto&& i : in) {
            auto size = impl::csv::read_matrix_size(i);
            auto padded_size = PADDING::total_size(size);
            sizes.push_back(size);
            padded_sizes.push_back(padded_size);
            total_data_size += padded_size.area();

            if (sizes.size() > 0 && sizes[0] != size) {
                throw std::runtime_error{"Data array contains matrices of different sizes"};
            }
        }


        std::vector<T, ALLOC> data(total_data_size);
        auto mat_data = data.data();
        for (dsize_t i = 0; i < in.size(); ++i) {
            impl::csv::read_data<T, PADDING>(in[i], sizes[i], padded_sizes[i], mat_data);
            mat_data += padded_sizes[i].area();
        }

        return data_array<T, ALLOC>{(dsize_t)sizes.size(), padded_sizes[0], std::move(data)};
    }

    template<typename PADDING>
    static data_array<T, ALLOC> load_from_csv(std::ifstream& in) {
        auto num_matrices = impl::csv::read_num_matrices(in);
        auto matrix_size = impl::csv::read_matrix_size(in);

        auto padded_matrix_size = PADDING::total_size(matrix_size);
        auto total_data_size = padded_matrix_size.area() * num_matrices;

        std::vector<T, ALLOC> data(total_data_size);
        auto mat_data = data.data();
        for (dsize_t i = 0; i < num_matrices; ++i) {
            impl::csv::read_data<T, PADDING>(in, matrix_size, padded_matrix_size, mat_data);
            mat_data += padded_matrix_size.area();
        }
        return data_array<T, ALLOC>{num_matrices, padded_matrix_size, std::move(data)};
    }

    void store_to_csv(std::vector<std::ofstream> outputs) const {
        for (dsize_t i = 0; i < outputs.size(); ++i) {
            impl::csv::write_matrix_size(outputs[i], matrix_size_);
            impl::csv::write_data(outputs[i], matrix_size_, data_.data() + matrix_size_.area() * i);
        }
    }

    void store_to_csv(std::ofstream& output) const {
        impl::csv::write_num_matrices(output, num_matrices_);

        for (dsize_t i = 0; i < num_matrices_; ++i) {
            impl::csv::write_matrix_size(output, matrix_size_);
            impl::csv::write_data(output, matrix_size_, data_.data() + matrix_size_.area() * i);
        }
    }

    size_type size() const {
        return num_matrices_ * matrix_size_.area();
    }

    size_type num_matrices() const {
        return num_matrices_;
    }

    dsize2_t matrix_size() const {
        return matrix_size_;
    }

    pointer data() {
        return data_.data();
    }

    const_pointer data() const {
        return data_.data();
    }

    iterator begin() {
        return data();
    }

    const_iterator begin() const {
        return data();
    }

    iterator end() {
        return data() + size();
    }

    const_iterator end() const {
        return data() + size();
    }

    matrix_view<T> view(dsize_t matrix_index) {
        return matrix_view<T>{
            matrix_size_,
            data_.data() + matrix_size_.area() * matrix_index
        };
    }

    const_matrix_view<T> view(dsize_t matrix_index) const {
        return const_matrix_view<T>{
            matrix_size_,
            data_.data() + matrix_size_.area() * matrix_index
        };
    }

private:
    dsize_t num_matrices_;
    dsize2_t matrix_size_;

    std::vector<T, ALLOC> data_;

    data_array(dsize_t num_matrices, dsize2_t matrix_size, std::vector<T, ALLOC>&& data)
        :num_matrices_(num_matrices), matrix_size_(matrix_size), data_(std::move(data))
    { }
};

}