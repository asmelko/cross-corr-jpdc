#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include "shared_mem.cuh"

namespace cg = cooperative_groups;

namespace cross {

template<typename T>
class row_slice_base {
public:
    __device__ row_slice_base(dsize_t start, dsize_t size)
        :start_(start), size_(size)
    {

    }

    __device__ dsize_t start_offset() const {
        return start_;
    }

    __device__ dsize_t size() const {
        return size_;
    }
protected:
    // Index the slice_data_ points to in the given row
    dsize_t start_;
    dsize_t size_;
};

template<typename T>
class row_slice: public row_slice_base<T>{
public:

    __device__ static row_slice<T> from_positions(dsize_t start, dsize_t end, T* src_data){
        return row_slice<T>{start, end - start, src_data};
    }

    __device__ static row_slice<T> from_position_size(dsize_t start, dsize_t size, T* src_data){
        return row_slice<T>{start, size, src_data};
    }

    __device__ row_slice()
        :row_slice_base<T>(0, 0), slice_data_(nullptr)
    {

    }

    __device__ row_slice(dsize_t start, dsize_t size, T* src_data)
        :row_slice_base<T>(start, size), slice_data_(src_data + start)
    {

    }

    __device__ T& operator [](dsize_t i) {
        return slice_data_[i];
    }

    __device__ const T& operator [](dsize_t i) const {
        return slice_data_[i];
    }

    __device__ const T* data() {
        return slice_data_;
    }

    __device__ row_slice<T> subslice(dsize_t size, dsize_t start_shift = 0) const {
        return row_slice<T>{
            start_ + start_shift,
            min(size, clamp_to_nonnegative((int)size_ - (int)start_shift)),
            slice_data_ - start_
        };
    }
private:
    T* slice_data_;
};

template<typename T>
class const_row_slice: public row_slice_base<T> {
public:

    __device__ static const_row_slice<T> from_positions(dsize_t start, dsize_t end, const T* src_data){
        return const_row_slice<T>{start, end - start, src_data};
    }

    __device__ static const_row_slice<T> from_position_size(dsize_t start, dsize_t size, const T* src_data){
        return const_row_slice<T>{start, size, src_data};
    }

    __device__ const_row_slice()
        :row_slice_base<T>(0, 0), slice_data_(nullptr)
    {

    }

    __device__ const_row_slice(dsize_t start, dsize_t size, const T* src_data)
        :row_slice_base<T>(start, size), slice_data_(src_data + start)
    {

    }

    __device__ const T& operator [](dsize_t i) const {
        return slice_data_[i];
    }

    __device__ const T* data() const {
        return slice_data_;
    }

    __device__ const_row_slice<T> subslice(dsize_t size, dsize_t start_shift = 0) const {
        return const_row_slice<T>{
            start_ + start_shift,
            min(size, clamp_to_nonnegative((int)size_ - (int)start_shift)),
            slice_data_ - start_
        };
    }

private:
    const T* slice_data_;
};

template<typename T>
class matrix_slice_base {
public:
    __device__ matrix_slice_base(dsize2_t top_left, dsize2_t size, dsize_t src_row_size)
        :top_left_(top_left), size_(size), src_row_size_(src_row_size)
    {

    }

    __device__ dsize2_t top_left() const {
        return top_left_;
    }

    __device__ dsize2_t size() const {
        return size_;
    }

    __device__ dsize2_t get_bottom_right() const {
        return top_left_ + size_;
    }

    __device__ dsize_t begin_x_src_idx() const {
        return top_left_.x;
    }

    __device__ dsize_t end_x_src_idx() const {
        return top_left_.x + size_.x;
    }

    __device__ dsize_t begin_y_src_idx() const {
        return top_left_.y;
    }

    __device__ dsize_t end_y_src_idx() const {
        return top_left_.y + size_.y;
    }
protected:
    dsize2_t top_left_;
    dsize2_t size_;
    dsize_t src_row_size_;
};

// TODO: Use template magic to merge const matrix slice and matrix slice
template<typename T>
class matrix_slice: public matrix_slice_base<T> {
public:
    __device__ static matrix_slice<T> from_positions(dsize2_t top_left, dsize2_t bottom_right, dsize_t src_row_size, T* src_data) {
        return matrix_slice<T>{top_left, bottom_right - top_left, src_row_size, src_data};
    }

    __device__ static matrix_slice<T> from_position_size(dsize2_t top_left, dsize2_t size, dsize_t src_row_size, T* src_data) {
        return matrix_slice<T>{top_left, size, src_row_size, src_data};
    }

    __device__ matrix_slice(dsize2_t top_left, dsize2_t size, dsize_t src_row_size, T* src_data)
        :matrix_slice_base<T>(top_left, size, src_row_size), src_data_(src_data)
    {

    }

    __device__ row_slice<T> row(dsize_t row) {
        return row_slice<T>::from_position_size(
            top_left.x,
            size.x,
            src_data_ + (row + top_left.y) * src_row_size
        );
    }

    __device__ const matrix_slice<T> submatrix_from_pos(
        dsize_t top_left_x,
        dsize_t top_left_y,
        dsize_t bottom_right_x,
        dsize_t bottom_right_y
    ) const {

    }
private:
    T* src_data_;
};

template<typename T>
struct const_matrix_slice: public matrix_slice_base<T> {
public:
    __device__ static const_matrix_slice<T> from_positions(dsize2_t top_left, dsize2_t bottom_right, dsize_t src_row_size, const T* src_data) {
        return const_matrix_slice<T>{top_left, bottom_right - top_left, src_row_size, src_data};
    }

    __device__ static const_matrix_slice<T> from_position_size(dsize2_t top_left, dsize2_t size, dsize_t src_row_size, const T* src_data) {
        return const_matrix_slice<T>{top_left, size, src_row_size, src_data};
    }

    __device__ const_matrix_slice(dsize2_t top_left, dsize2_t size, dsize_t src_row_size, const T* src_data)
        :matrix_slice_base<T>(top_left, size, src_row_size), src_data_(src_data)
    {

    }

    __device__ const_row_slice<T> row(dsize_t row) {
        return const_row_slice<T>::from_position_size(
            top_left_.x,
            size_.x,
            src_data_ + (row + top_left_.y) * src_row_size_
        );
    }

    __device__ const const_matrix_slice<T> submatrix_from_pos(
        dsize_t top_left_x,
        dsize_t top_left_y,
        dsize_t bottom_right_x,
        dsize_t bottom_right_y
    ) const {

    }
private:
    const T* src_data_;
};

template<typename T>
class row_slice_buffer {
public:

    using value_type = T;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    __device__ row_slice_buffer(const shared_mem_buffer<T>& buffer)
        :buffer_(buffer)
    {

    }

    __device__ reference operator [](dsize_t i) {
        return buffer_[i];
    }

    __device__ const_reference operator [](dsize_t i) const {
        return buffer_[i];
    }

    __device__ size_type allocated_size() const {
        return buffer_.size();
    }

    __device__ size_type size() const {
        return slice_.size();
    }

    __device__ size_type load(cg::thread_block ctb, row_slice<T>&& slice) {
        // TODO: Assert slice size <= buffer size
        slice_ = std::move(slice);
        return buffer_.load(ctb, slice_.data(), slice_.size());
    }

    __device__ size_type start_offset() const {
        return slice_.start_offset();
    }
private:
    shared_mem_buffer<T> buffer_;
    row_slice<T> slice_;
};


template<typename T, dsize_t NUM_PARTS>
class row_ring_buffer{
public:

    using value_type = T;
    using size_type = dsize_t;
    using reference = value_type&;
    using const_reference = const value_type&;

    __device__ row_ring_buffer(
        cg::thread_block ctb,
        row_slice<T> src,
        shared_mem_buffer<T> buffer
    )
        :src_(src), buffer_(buffer), tail_(0), tail_src_offset_(0)
    {
        dsize_t load_size = load_part(ctb, 0, 0);
        load_part(ctb, load_size, load_size);

        // Don't need to track the number of loaded values, as if we have
        // less than buffer_size, the following loads will load zero values,
        // as there are non left in the source
        // At most the tail_src_offset_ + buffer_.size() will be outside
        // the matrix, which is caught by clamping in the load method and clamped to 0
    }

    __device__ dsize_t load_next_parts(cg::thread_block ctb, dsize_t num_parts) {
        dsize_t load_size = 0;
        for (dsize_t i = 0; i < num_parts; ++i) {
            load_size += load_next(ctb);
        }
        return load_size;
    }

    __device__ size_t load_next(cg::thread_block ctb) {
        dsize_t load_size = load_part(
            ctb,
            tail_,
            tail_src_offset_ + buffer_.size()
        );

        tail_ = (tail_ + load_size) % buffer_.size();
        tail_src_offset_ += load_size;

        // if ((ctb.group_index().x == 2 || ctb.group_index().x == 1) && ctb.group_index().y == 0 && ctb.thread_rank() == 0) {
        //     printf("Block: %u, Src: [%u, %u], Size: %u, Tail: %u, Src offset: %u\n", ctb.group_index().x, src_.start_offset(), src_.size(), load_size, tail_, tail_src_offset_);
        // }

        return load_size;
    }


    __device__ size_type start_offset() const {
        return src_.start_offset() + tail_src_offset_;
    }

    __device__ reference operator [](dsize_t i) {
        return buffer_[(tail_ + i) % buffer_.size()];
    }

    __device__ const_reference operator [](dsize_t i) const {
        return buffer_[(tail_ + i) % buffer_.size()];
    }

    __device__ size_type size() const {
        return buffer_.size();
    }

    __device__ size_type num_loaded() const {
        return min(buffer_.size(), src_.size() - tail_src_offset_);
    }
private:
    row_slice<T> src_;
    shared_mem_buffer<T> buffer_;
    size_type tail_;
    size_type tail_src_offset_;

    __device__ dsize_t load_part(cg::thread_block ctb, dsize_t buffer_offset, dsize_t src_offset) {
        auto part_slice = src_.subslice(buffer_.size() / NUM_PARTS, src_offset);
        return buffer_.load(
            ctb,
            part_slice.data(),
            min(buffer_.size() / NUM_PARTS, part_slice.size(),
            buffer_offset
        );
    }
};


}