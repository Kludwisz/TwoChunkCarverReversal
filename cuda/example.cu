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
    /*
    seed 4382749480380    at (x=471345, z=1218423)
    seed 276921524333971    at (x=-1767353, z=-1255334)
    seed 127457682554596    at (x=932587, z=1077254)
    seed 276921524333971    at (x=-1767352, z=-1255334)
    seed 224668638317660    at (x=-1712945, z=-1711638)
    seed 4382749480380    at (x=471344, z=1218423)
    seed 95572529922801    at (x=304381, z=93704)
    seed 24750512097244    at (x=-1234342, z=1663550)
    seed 4382749480380    at (x=471348, z=1218423)
    */
}