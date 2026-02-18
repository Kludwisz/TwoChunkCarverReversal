#include "two_carver_reversal.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


static void cuda_error(const char* file, int line_no, const char* msg) {
    printf("Line %d, file %s:\nCUDA ERROR - %s\n", line_no, file, msg);
    exit(1);
}
#define CUDA_CHECK(code) do {\
    code;\
    cudaError_t cuda_status = cudaGetLastError();\
    if (cuda_status != cudaSuccess) {\
        cuda_error(__FILE__, __LINE__, cudaGetErrorName(cuda_status));\
    }\
} while(0)


namespace gputcr {
    constexpr int32_t CHUNKS_ON_AXIS = 60'000'000 / 16;
    constexpr int32_t HALF_CHUNKS = CHUNKS_ON_AXIS / 2;
    //constexpr uint64_t MOD_48 = 1ULL << 48; 
    constexpr uint64_t MASK_48 = (1ULL << 48) - 1; 
    constexpr uint64_t LCG_A = 0x5deece66d;
    constexpr uint32_t LCG_B = 11;
    // -------------------------------------------------------

    constexpr uint32_t THREADS_PER_BLOCK = 256;
    constexpr uint32_t MAX_INPUTS_PER_RUN = 512;
    __constant__ uint64_t carver1_vec[MAX_INPUTS_PER_RUN];
    __constant__ uint64_t carver2_vec[MAX_INPUTS_PER_RUN];
    __constant__ int32_t full_offset_xz;

    constexpr uint32_t MAX_RESULTS_PER_RUN = 64 * 1024;
    __managed__ Result managed_result_vec[MAX_RESULTS_PER_RUN];
    __managed__ uint32_t managed_result_count;

    // Hensel lifting for Nx ^ Mx = C
    __device__ static uint64_t hensel_lift_gpu_64_fast(int32_t n, int32_t m, uint64_t target) {
        
        uint64_t pa = 0;
        for (int bits = 1; bits <= 48; bits++) {
            const uint64_t mask = (1ULL << bits) - 1; 
            const uint64_t lhs_0 = (n*pa ^ m*pa) & mask;
            pa |= (lhs_0 == (target & mask) ? 0 : (1ULL << bits-1));
            // not checking the other branch at all - makes this incorrect in many cases but keeps
            // the correct results and seems to be 
        }
        return pa & MASK_48;
    }

    // ------------------------------------------------------
    // NextLongReverser by Matthew Bolan translated to CUDA
    // original code: 
    // https://github.com/SeedFinding/NextLongReverser
    // ------------------------------------------------------

    __device__ static inline int64_t floorDiv(int64_t x, int64_t y) {
        const int64_t q = x / y;
        if ((x ^ y) < 0 && (q * y != x)) {
            return q - 1;
        }
        return q;
    }

    __device__ static inline int reverse_nextLong(uint64_t nextLong_lower48, uint64_t output[]) {
        int seedID = 0;

        int64_t lowerBits = nextLong_lower48 & 0xffff'ffffULL;
        int64_t upperBits = nextLong_lower48 >> 32;
        //Did the lower bits affect the upper bits
        if ((lowerBits & 0x80000000LL) != 0)
            upperBits += 1; //restoring the initial value of the upper bits

        //TODO I can only guarantee the algorithm's correctness for bitsOfDanger = 0 but believe 1 should still always work, needs to be confirmed!!!

        //The algorithm is meant to have bitsOfDanger = 0, but this runs into overflow issues.
        //By using a different small value, we introduce small numerical error which probably cannot break things
        //while keeping everything in range of a long and avoiding nasty BigDecimal/BigInteger overhead
        int bitsOfDanger = 1;

        int64_t lowMin = lowerBits << 16 - bitsOfDanger;
        int64_t lowMax = ((lowerBits + 1) << 16 - bitsOfDanger) - 1;
        int64_t upperMin = ((upperBits << 16) - 107048004364969LL) >> bitsOfDanger;

        //hardcoded matrix multiplication again
        int64_t m1lv = floorDiv(lowMax * -33441LL + upperMin * 17549LL, 1LL << 31 - bitsOfDanger) + 1; //I cancelled out a common factor of 2 in this line
        int64_t m2lv = floorDiv(lowMin * 46603LL + upperMin * 39761LL, 1LL << 32 - bitsOfDanger) + 1;

        int64_t seed;

        // (0,0) -> 0.6003265380859375
        seed = (-39761LL * m1lv + 35098LL * m2lv);
        if ((46603LL * m1lv + 66882LL * m2lv) + 107048004364969LL >> 16 == upperBits) {
            if (((uint64_t)seed >> 16) == lowerBits)
                output[seedID++] = ((254681119335897ULL * (uint64_t)seed + 120305458776662ULL) & MASK_48); //pull back 2 LCG calls
        }
        //(1,0) -> 0.282440185546875
        seed = (-39761LL * (m1lv + 1) + 35098LL * m2lv);
        if ((46603LL * (m1lv + 1) + 66882LL * m2lv) + 107048004364969LL >> 16 == upperBits) {
            if (((uint64_t)seed >> 16) == lowerBits)
                output[seedID++] = ((254681119335897ULL * (uint64_t)seed + 120305458776662ULL) & MASK_48); //pull back 2 LCG calls
        }
        //(0,1) -> 0.1172332763671875
        seed = (-39761LL * m1lv + 35098LL * (m2lv + 1));
        if ((46603LL * m1lv + 66882LL * (m2lv + 1)) + 107048004364969LL >> 16 == upperBits) {
            if (((uint64_t)seed >> 16) == lowerBits)
                output[seedID++] = ((254681119335897ULL * (uint64_t)seed + 120305458776662ULL) & MASK_48); //pull back 2 LCG calls
        }
        //(1,1) -> 0.0

        return seedID;
    }

    // End of NextLongReverser
    // ------------------------------------------------------

    // modInverse implementation taken from Neil's and KaptainWutax's mc_math_java library (thank you!)
    // https://github.com/SeedFinding/mc_math_java/blob/main/src/main/java/com/seedfinding/mcmath/util/Mth.java
    __device__ static inline uint64_t modinv(uint64_t value, int bits) {
        uint64_t x = ((((value << 1) ^ value) & 4) << 1) ^ value;
        x *= 2 - value * x;
        x *= 2 - value * x;
        x *= 2 - value * x;
        x *= 2 - value * x;
        return x & ((1ULL << bits) - 1);
    }

    // Java Random utils
    __device__ static inline uint64_t advance1(uint64_t seed) {
        return (seed * LCG_A + LCG_B) & MASK_48;
    }
    __device__ static inline uint64_t advance2(uint64_t seed) {
        return (seed * 205749139540585ULL + 277363943098ULL) & MASK_48;
    }
    __device__ static inline uint64_t back2(uint64_t seed) {
        return (seed * 254681119335897ULL + 120305458776662ULL) & MASK_48;
    }
    __device__ static inline int64_t nextLong(uint64_t seed) {
        int64_t high = (advance1(seed) >> 16) << 32;
        int64_t low = static_cast<int64_t>(static_cast<int32_t>(advance2(seed) >> 16));
        return high + low;
    }

    // ------------------------------------------------------

    __global__ void two_carver_reversal_kernel_x() {
        const int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        const uint64_t carver1 = carver1_vec[blockIdx.y];
        const uint64_t carver2 = carver2_vec[blockIdx.z];
        const int32_t x1 = tid - HALF_CHUNKS;
        const int32_t x2 = tid + full_offset_xz;
        
        uint64_t a = hensel_lift_gpu_64_fast(x1, x2, carver1^carver2);

        uint64_t iseeds[2];
        int iseeds_count = reverse_nextLong(a & MASK_48, iseeds);

        #pragma unroll
        for (int i = 0; i < 2 && i < iseeds_count; i++)  {
            // if iseed is for x no need to go back
            uint64_t iseed = iseeds[i];
            uint64_t structure_seed = (iseed ^ LCG_A) & MASK_48;
            uint64_t iseed_z = advance2(iseed); // +2

            uint64_t b = nextLong(iseed_z) & MASK_48;
            uint64_t r = (carver1 ^ a*x1 ^ structure_seed) & MASK_48;
            int tz_b = __ffsll(b) - 1;
            uint64_t mod = 1ULL << 48-tz_b;
            b >>= tz_b;
            r >>= tz_b; // dont care if it's incorrect

            uint64_t binv = modinv(b, 48-tz_b);
            int64_t z_candidate = static_cast<int64_t>((r * binv) & (mod-1));
            if (z_candidate >= (mod >> 1)) {
                z_candidate -= mod; // offset to reduce absolute value
            }

            if (std::abs(z_candidate) <= HALF_CHUNKS) {
                int32_t z = static_cast<int32_t>(z_candidate);
                uint64_t result_carver1 = a*x1 ^ b*z ^ structure_seed;
                uint64_t result_carver2 = a*x2 ^ b*z ^ structure_seed;
                if ((result_carver1 & MASK_48) == (carver1 & MASK_48) && (result_carver2 & MASK_48) == (carver2 & MASK_48)) {
                    uint32_t idx = atomicAdd(&managed_result_count, 1);
                    if (idx < MAX_RESULTS_PER_RUN) {
                        managed_result_vec[idx] = {structure_seed, x1, z};
                    }
                }
            }
        }
    }

    __global__ void two_carver_reversal_kernel_z() {
        const int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        const uint64_t carver1 = carver1_vec[blockIdx.y];
        const uint64_t carver2 = carver2_vec[blockIdx.z];
        const int32_t z1 = tid - HALF_CHUNKS;
        const int32_t z2 = tid + full_offset_xz;
        
        uint64_t b = hensel_lift_gpu_64_fast(z1, z2, carver1^carver2);

        uint64_t iseeds[2];
        int iseeds_count = reverse_nextLong(b & MASK_48, iseeds);

        #pragma unroll
        for (int i = 0; i < 2 && i < iseeds_count; i++)  {
            // iseed is for z, need to go back to the x call starting state
            uint64_t iseed = iseeds[i];
            iseed = back2(iseed);
            uint64_t structure_seed = (iseed ^ LCG_A) & MASK_48;
            uint64_t iseed_x = iseed;

            uint64_t a = nextLong(iseed_x) & MASK_48;
            uint64_t r = (carver1 ^ b*z1 ^ structure_seed) & MASK_48;
            int tz_a = __ffsll(a) - 1;
            uint64_t mod = 1ULL << 48-tz_a;
            a >>= tz_a;
            r >>= tz_a; // dont care if incorrect

            uint64_t ainv = modinv(a, 48-tz_a);
            int64_t x_candidate = static_cast<int64_t>((r * ainv) & (mod-1));
            if (x_candidate >= (mod>>1)) {
                x_candidate -= mod; // offset to reduce absolute value
            }

            if (std::abs(x_candidate) <= HALF_CHUNKS) {
                int32_t x = static_cast<int32_t>(x_candidate);

                uint64_t result_carver1 = a*x ^ b*z1 ^ structure_seed;
                uint64_t result_carver2 = a*x ^ b*z2 ^ structure_seed;
                if ((result_carver1 & MASK_48) == (carver1 & MASK_48) && (result_carver2 & MASK_48) == (carver2 & MASK_48)) {
                    uint32_t idx = atomicAdd(&managed_result_count, 1);
                    if (idx < MAX_RESULTS_PER_RUN) {
                        managed_result_vec[idx] = {structure_seed, x, z1};
                    }
                }
            }
        }
    }

    __host__ static void run_kernel(
        const ChunkOffset chunk_offset, 
        const uint64_t* data1, uint32_t num_elements_data1,
        const uint64_t* data2, uint32_t num_elements_data2,
        std::vector<Result>& results
    ) {
        if (chunk_offset.x != 0 && chunk_offset.z != 0) {
            return;
        }
        if (chunk_offset.x == 0 && chunk_offset.z == 0) {
            return;
        }
        managed_result_count = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(carver1_vec, (void*)data1, num_elements_data1 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(carver2_vec, (void*)data2, num_elements_data2 * sizeof(uint64_t)));

        uint32_t grid_x = (CHUNKS_ON_AXIS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        dim3 blockConfig(THREADS_PER_BLOCK, 1, 1);
        dim3 gridConfig(grid_x, num_elements_data1, num_elements_data2);

        if (chunk_offset.x != 0) {
            //printf("two_carver_reversal_kernel_x : grid_x=%u, grid_y=%u, grid_z=%u\n", gridConfig.x, gridConfig.y, gridConfig.z);
            two_carver_reversal_kernel_x<<<gridConfig, blockConfig>>>();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        else if (chunk_offset.z != 0) {
            //printf("two_carver_reversal_kernel_z : grid_x=%u, grid_y=%u, grid_z=%u\n", gridConfig.x, gridConfig.y, gridConfig.z);
            two_carver_reversal_kernel_z<<<gridConfig, blockConfig>>>();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // read results from managed buffer
        uint32_t n = managed_result_count;
        for (uint32_t i = 0; i < n; i++) {
            results.push_back(managed_result_vec[i]);
        }
    }

    void reverse_carver_seed_pairs_gpu(
        const ChunkOffset chunk_offset,
        const std::vector<uint64_t>& carver_seeds_chunk_1,
        const std::vector<uint64_t>& carver_seeds_chunk_2,
        std::vector<Result>& combined_results, int device_id
    ) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (chunk_offset.x != 0) {
            int32_t real_offset = chunk_offset.x - HALF_CHUNKS;
            CUDA_CHECK(cudaMemcpyToSymbol(full_offset_xz, (void*)(&real_offset), sizeof(int32_t)));
        }
        else if (chunk_offset.z != 0) {
            int32_t real_offset = chunk_offset.z - HALF_CHUNKS;
            CUDA_CHECK(cudaMemcpyToSymbol(full_offset_xz, (void*)(&real_offset), sizeof(int32_t)));
        }

        const uint32_t s1 = static_cast<uint32_t>(carver_seeds_chunk_1.size());
        const uint32_t s2 = static_cast<uint32_t>(carver_seeds_chunk_2.size());
        const uint32_t carver1_runs = (s1 + MAX_INPUTS_PER_RUN - 1) / MAX_INPUTS_PER_RUN;
        const uint32_t carver2_runs = (s2 + MAX_INPUTS_PER_RUN - 1) / MAX_INPUTS_PER_RUN;

        for (uint32_t rc1 = 0; rc1 < carver1_runs; rc1++) {
            for (uint32_t rc2 = 0; rc2 < carver2_runs; rc2++) {
                //printf("rc1=%d rc2=%d\n", rc1, rc2);
                uint32_t start1 = rc1*MAX_INPUTS_PER_RUN;
                uint32_t end1 = std::min(start1 + MAX_INPUTS_PER_RUN, s1);
                uint32_t size1 = end1 - start1;
                uint32_t start2 = rc2*MAX_INPUTS_PER_RUN;
                uint32_t end2 = std::min(start2 + MAX_INPUTS_PER_RUN, s2);
                uint32_t size2 = end2 - start2;

                const uint64_t* data1 = &(carver_seeds_chunk_1.data()[start1]);
                const uint64_t* data2 = &(carver_seeds_chunk_2.data()[start2]);

                run_kernel(chunk_offset, data1, size1, data2, size2, combined_results);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaDeviceReset());
    }
};