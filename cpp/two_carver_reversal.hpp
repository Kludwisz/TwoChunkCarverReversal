#ifndef TWO_CARVER_REVERSAL
#define TWO_CARVER_REVERSAL

#include <vector>
#include <cstdint> 

namespace tcr {
    struct Result {
        uint64_t structure_seed;
        int32_t x;
        int32_t z;
    };

    uint64_t get_carver_seed(uint64_t structure_seed, int32_t x, int32_t z);
    void reverse_given_x(uint64_t carver1, uint64_t carver2, int32_t x1, int32_t x2, std::vector<Result>& results, bool accurate_lifting = false);
    void reverse_given_z(uint64_t carver1, uint64_t carver2, int32_t z1, int32_t z2, std::vector<Result>& results, bool accurate_lifting = false);
    void reverse_carver_seed_pair(uint64_t carver1, uint64_t carver2, int32_t offset_x, int32_t offset_z, std::vector<Result>& results, bool accurate_lifting = false);
};

#endif
