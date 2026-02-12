/*
compile with:
g++ example.cpp two_carver_reversal.cpp -O3 -o example
run with:
./example
*/

#include <cstdio>
#include "two_carver_reversal.hpp"


int main() {
    const int base = 420420;
    for (uint64_t c1 = base; c1 <= base+5; c1++) {
        for (uint64_t c2 = base; c2 <= base+5; c2++) {
            printf("-- carver seed #1 = %llu, carver seed #2 = %llu, offset = (1,0)\n", c1, c2);
            std::vector<tcr::Result> results;
            tcr::reverse_carver_seed_pair(c1, c2, 1, 0, results);
            for (auto res : results) {
                printf("Got a result! structure seed: %llu  at %d,%d ; %d,%d\n", res.structure_seed, res.x, res.z, res.x+1, res.z);
            }
        }
    }

    return 0;
}