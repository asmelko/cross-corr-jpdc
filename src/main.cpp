#include <iostream>
#include <fstream>

#include "matrix.hpp"
#include "cross_corr.hpp"
#include "allocator.cuh"

using namespace cross;

using mat = matrix<float, pinned_allocator<float>>;

void load_inputs(const char* ref_path, const char* target_path, mat& out_ref, mat& out_target) {
    std::ifstream ref_file(ref_path);
    std::ifstream target_file(target_path);

    out_ref = mat::load_from_csv(ref_file);
    out_target = mat::load_from_csv(target_file);
}

int main(int argc, char **argv) {
    // TODO: Better argument parsing
    if (argc != 4) {
        std::cerr << "Invalid number of arguments, expected " << 4 << " , got " << argc - 1 << "\n";
        std::cerr << "Usage: " << argv[0] << " <ref_path> <target_path> <out_path>\n";
        return 1;
    }
    try {
        mat ref;
        mat target;
        load_inputs(argv[1], argv[2], ref, target);


        naive<mat> alg{std::move(ref), std::move(target)};
        alg.prepare();

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