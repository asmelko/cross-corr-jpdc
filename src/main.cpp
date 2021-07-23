#include <iostream>
#include <fstream>

#include "matrix.hpp"
#include "cross_corr.hpp"
#include "allocator.cuh"

using namespace cross;

using mat = matrix<float, pinned_allocator<float>>;

int main(int argc, char **argv) {
    // TODO: Better argument parsing
    if (argc != 4) {
        std::cerr << "Invalid number of arguments, expected " << 4 << " , got " << argc - 1 << "\n";
        std::cerr << "Usage: " << argv[0] << " <ref_path> <target_path> <out_path>\n";
        return 1;
    }
    try {


        //naive_original_alg<mat> alg{std::move(ref), std::move(target)};
        fft_original_alg<mat, true> alg;
        alg.prepare(argv[1], argv[2]);

        alg.run();

        alg.finalize();

        auto res = alg.results();

        std::ofstream out_file(argv[3]);
        res.store_to_csv(out_file);
    }
    catch (std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }
}