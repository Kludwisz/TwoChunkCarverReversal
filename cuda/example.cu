/*
compile with:
nvcc example.cu two_carver_reversal.cu -o example
run with:
./example
*/

#include <cstdio>
#include "two_carver_reversal.cuh"


int main() {
    const int base = 420420;
    std::vector<uint64_t> carvers1;
    std::vector<uint64_t> carvers2;
    for (uint64_t c = base; c <= base+5; c++) {
        carvers1.push_back(c);
        carvers2.push_back(c);
    }

    std::vector<gputcr::Result> results;
    gputcr::reverse_carver_seed_pairs_gpu(gputcr::ChunkOffset(1, 0), carvers1, carvers2, results);

    for (auto& res : results) {
        printf("seed %lld    at (x1=%d, z1=%d)\n", res.structure_seed, res.chunk_x, res.chunk_z);
    }
}