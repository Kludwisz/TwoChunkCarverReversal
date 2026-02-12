#ifndef TWO_CARVER_REVERSAL_CUH
#define TWO_CARVER_REVERSAL_CUH

#include <cstdint>
#include <vector>


namespace gputcr {
    struct ChunkOffset {
        int32_t x;
        int32_t z;

        inline ChunkOffset(int32_t x, int32_t z) : x(x), z(z) {}
    };

    struct Result {
        uint64_t structure_seed;
        int32_t chunk_x;
        int32_t chunk_z;
    };

    void reverse_carver_seed_pairs_gpu(
        const ChunkOffset chunk_offset,
        const std::vector<uint64_t>& carver_seeds_chunk_1,
        const std::vector<uint64_t>& carver_seeds_chunk_2,
        std::vector<Result>& combined_results, int device_id = 0
    );
}

#endif