/*
compile with:
nvcc example.cu two_carver_reversal.cu -o example
run with:
./example
*/

#include <cstdio>
#include <chrono>
#include "two_carver_reversal.cuh"


int main() {
    const int base = 420420;
    std::vector<uint64_t> carvers1;
    std::vector<uint64_t> carvers2;
    for (uint64_t c = base; c <= base+20; c++) {
        carvers1.push_back(c);
        carvers2.push_back(c);
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<gputcr::Result> results;
    gputcr::reverse_carver_seed_pairs_gpu(gputcr::ChunkOffset(1, 0), carvers1, carvers2, results);
    auto t1 = std::chrono::steady_clock::now();

    for (auto& res : results) {
        printf("seed %lld    at (x=%d, z=%d)\n", res.structure_seed, res.chunk_x, res.chunk_z);
    }

    printf("Kernel took %f ms", (t1-t0).count() * 1e-6);

    // Expected output:
    // seed 127457682554596    at (x=932587, z=1077254)
}