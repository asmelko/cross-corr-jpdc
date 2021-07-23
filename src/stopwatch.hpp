#pragma once

#include <vector>
#include <chrono>
#include <string>

#include <cuda_runtime.h>

namespace cross {

#define CUDA_MEASURE(label, block)      \
    do {                                \
        this->sw_.cuda_insert_start();  \
        block;                          \
        this->sw_.cuda_insert_stop();   \
        this->sw_.cuda_measure(label);  \
    } while(0)

#define CPU_MEASURE(label, block)       \
    do {                                \
        this->sw_.cpu_start();          \
        block;                          \
        this->sw_.cpu_measure(label);   \
    } while(0)

template<typename CLOCK>
class StopWatch {
public:
    StopWatch(std::size_t num_measurements)
        :measurements_(num_measurements)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~StopWatch() {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }

    void cpu_start() {
        start_ = CLOCK::now();
    }

    void cpu_measure(std::size_t label) {
        measurements_[label] = CLOCK::now() - start_;
    }

    void cuda_insert_start() {
        cudaEventRecord(start);
    }

    void cuda_insert_stop() {
        cudaEventRecord(stop);
    }

    void cuda_measure(std::size_t label) {
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        measurements_[label] = std::chrono::duration_cast<typename CLOCK::duration>(std::chrono::duration<double, std::milli>(milliseconds));
    }

    void store_measurement(std::size_t label, typename CLOCK::duration measurement) {
        measurements_[label] = measurement;
    }

    const std::vector<typename CLOCK::duration> results() const {
        return measurements_;
    }
private:
    typename CLOCK::time_point start_;
    std::vector<typename CLOCK::duration> measurements_;

    cudaEvent_t start, stop;

};

}