#include <cstdio> 
#include <vector> 
#include <chrono>
#include <cmath>

#include "two_carver_reversal.hpp"

// How carver seeding works

// Input: seed (the structure seed), x (x coordinate of the chunk), z (z coordinate of the chunk).
// Output: carver_seed (the carver seed for the given chunk)
// rand := java.util.Random(seed)
// a := rand.nextLong()
// b := rand.nextLong()
// carver_seed = lower_48_bits_of(a*x ^ b*z ^ seed)


// Algorithm explanation

// Let's assume the two carver chunks are at the same z coordinate. We have
//      ax     ^ bz ^ seed = C1  (mod 2**48)  [C1 = carver seed of first chunk]
//      a(x+D) ^ bz ^ seed = C2  (mod 2**48)  [C2 = carver seed of second chunk]
//
// Xoring the equations:
//      ax ^ a(x+D) = C1 ^ C2 (mod 2**48)
// 
// We can bruteforce all 3.75 million possible values of x. 
// For each such value we get an equation of the form
//      Ma ^ Na = C1 ^ C2  (mod 2**48)
// where a is the only unknown.
// This can be solved iteratively, by reconstructing a from the lowest bits upwards (see hensel_lift).
//
// In carver seeding, a is the value of a Java Random nextLong and can be reversed
// back to the Java Random internal state using Matthew Bolan's NextLongReverser code.
// That in turn gives us the structure seed, and all we're missing now is the z coordinate value.
// 
// To recover z, let's transform the first carver seed equation:
//      ax ^ bz ^ seed = C1  (mod 2**48)
//      bz = ax ^ seed ^ C1  (mod 2**48)
// Notice that everything on the right-hand-side is already known, let RHS = R.
//      bz = R  (mod 2**48)
// 
// If b is odd, we can directly calculate its inverse modulo 2**48 and multiply both sides by that.
// Otherwise, we need to eliminate all the factors of 2 first, which is possible if and only if 
// R has at least as many factors of 2 as b (if it's not possible the current x value yields no results). 
// Let p be the number of excluded factors of 2. Then:
//      (b >> p)z = (R >> p)  (mod 2**(48-p))
// and we can calculate the mod inverse of b>>p instead, giving
//      z = (R >> p) * modinv((b >> p), 2**(48-p))  (mod 2**(48-p))
// 
// This z value is under the reduced modulo, and we're targetting modulo 2**48.
// Therefore, we get 2**p valid solutions for z. Fortunately, since p is usually small, it's
// sufficient to calculate the z value nearest to 0 under the reduced mod, as it will be the
// only reasonable candidate under the original mod as well.
//
// Finally, we have the value of z mod 2**(48-p), and we can map it back to the actual chunk z coordinate
// by treating it as a signed U2 value stored on 48-p bits. That gives two valid ranges for z:
// [0, 1875000] and [2**(48-p) - 1875000, 2**(48-p)). If the z value falls into any of these, we get a result.
//
// The exact same process can be applied when starting from different z coordinates and the same x coordinate.
// In that case, the algorithm recovers b first, and calculates x in the final step.

namespace tcr {
    constexpr int32_t CHUNKS_ON_AXIS = 60'000'000 / 16;
    constexpr int32_t HALF_CHUNKS = CHUNKS_ON_AXIS / 2;
    constexpr uint64_t MOD_48 = 1ULL << 48; 
    constexpr uint64_t MASK_48 = (1ULL << 48) - 1; 
    constexpr uint64_t LCG_A = 0x5deece66d;
    constexpr uint32_t LCG_B = 11;

    // ----------------------------------------------------------------

    static void hensel_lift(uint64_t pa, int bits, int32_t n, int32_t m, uint64_t target, std::vector<uint64_t>& results) { 
        uint64_t mask = (1ULL << bits) - 1; 
        uint64_t pa0 = pa; 
        uint64_t pa1 = (pa | (1ULL << bits-1));
        uint64_t lhs_0 = (n*pa0 ^ m*pa0) & mask;
        uint64_t lhs_1 = (n*pa1 ^ m*pa1) & mask; 
        uint64_t target_masked = target & mask; 
        //printf("pa = %llu, target = %llu\n", pa, target); 
        //printf("lhs0 = %llu, target = %llu\n", lhs_0, target_masked); 
        //printf("lhs1 = %llu, target = %llu\n", lhs_1, target_masked); 

        if (lhs_0 == target_masked) { 
            if (bits == 48) { 
                results.push_back(pa0 & MASK_48); 
                return; 
            } 
            hensel_lift(pa0, bits+1, n, m, target, results); 
        } 
        if (lhs_1 == target_masked) { 
            if (bits == 48) { 
                results.push_back(pa1 & MASK_48); 
                return; 
            } 
            hensel_lift(pa1, bits+1, n, m, target, results); 
        } 
    } 

    static void hensel_lift_fast(int32_t n, int32_t m, uint64_t target, std::vector<uint64_t>& results) {
        uint64_t pa = 0;
        for (int bits = 1; bits <= 48; bits++) {
            const uint64_t mask = (1ULL << bits) - 1; 
            const uint64_t pa1 = (pa | (1ULL << bits-1));
            const uint64_t lhs_0 = (n*pa ^ m*pa) & mask;
            const uint64_t lhs_1 = (n*pa1 ^ m*pa1) & mask; 
            const uint64_t target_masked = target & mask;

            if (lhs_0 == target_masked) {
                continue; // "add" a 0 bit
            }
            else if (lhs_1 == target_masked) {
                pa = pa1;
                continue;
            }
            else {
                return;
            }
        }
        results.push_back(pa & MASK_48);
    }

    // -------------------------------------------------
    // NextLongReverser by Matthew Bolan translated to C
    // original code: 
    // https://github.com/SeedFinding/NextLongReverser
    // -------------------------------------------------

    static int64_t floorDiv(int64_t x, int64_t y) {
        const int64_t q = x / y;
        if ((x ^ y) < 0 && (q * y != x)) {
            return q - 1;
        }
        return q;
    }

    static int reverse_nextLong(uint64_t nextLong_lower48, uint64_t output[]) {
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

    // -------------------------------------------------

    // modInverse implementation taken from Neil's and KaptainWutax's mc_math_java library (thank you!)
    // https://github.com/SeedFinding/mc_math_java/blob/main/src/main/java/com/seedfinding/mcmath/util/Mth.java
    static uint64_t modinv(uint64_t value, int bits) {
        uint64_t x = ((((value << 1) ^ value) & 4) << 1) ^ value;
        x *= 2 - value * x;
        x *= 2 - value * x;
        x *= 2 - value * x;
        x *= 2 - value * x;
        return x & ((1ULL << bits) - 1);
    }

    static inline uint64_t advance1(uint64_t seed) {
        return (seed * LCG_A + LCG_B) & MASK_48;
    }
    static inline uint64_t advance2(uint64_t seed) {
        return (seed * 205749139540585 + 277363943098) & MASK_48;
    }
    static inline uint64_t back2(uint64_t seed) {
        return (seed * 254681119335897 + 120305458776662) & MASK_48;
    }

    static int64_t nextLong(uint64_t seed) {
        int64_t high = (advance1(seed) >> 16) << 32;
        int64_t low = static_cast<int64_t>(static_cast<int32_t>(advance2(seed) >> 16));
        return high + low;
    }

    uint64_t get_carver_seed(uint64_t structure_seed, int32_t x, int32_t z) {
        uint64_t s = (structure_seed ^ LCG_A) & MASK_48;
        uint64_t a = nextLong(s) & MASK_48;
        uint64_t b = nextLong(advance2(s)) & MASK_48;
        return (x*a ^ z*b ^ structure_seed) & MASK_48;
    }

    void reverse_given_x(uint64_t carver1, uint64_t carver2, int32_t x1, int32_t x2, std::vector<Result>& results, bool accurate_lifting = false) { 
        // xa ^ (x+d)a = C1 ^ C2 
        std::vector<uint64_t> a_values;
        if (accurate_lifting) {
            hensel_lift(0, 1, x1, x2, carver1 ^ carver2, a_values); 
        }
        else {
            hensel_lift_fast(x1, x2, carver1 ^ carver2, a_values);
        }

        for (auto a : a_values) {
            uint64_t iseeds[2];
            int iseeds_count = reverse_nextLong(a & MASK_48, iseeds);

            for (int i = 0; i < iseeds_count; i++)  {
                // if iseed is for x no need to go back
                uint64_t iseed = iseeds[i];
                uint64_t structure_seed = (iseed ^ LCG_A) & MASK_48;
                uint64_t iseed_z = advance2(iseed); // +2

                uint64_t b = nextLong(iseed_z) & MASK_48;
                uint64_t r = (carver1 ^ a*x1 ^ structure_seed) & MASK_48;
                int tz_b = __builtin_ctzll(b);
                int tz_r = __builtin_ctzll(b);
                if (tz_b > tz_r) {
                    continue; // can't do modinv
                }
                uint64_t mod = 1ULL << 48-tz_b;
                b >>= tz_b;
                r >>= tz_b;

                uint64_t binv = modinv(b, 48-tz_b);
                int64_t z_candidate = static_cast<int64_t>((r * binv) & (mod-1));
                if (z_candidate >= (mod >> 1)) {
                    z_candidate -= mod; // offset to reduce absolute value
                }

                if (std::abs(z_candidate) <= HALF_CHUNKS) {
                    int32_t z = static_cast<int32_t>(z_candidate);

                    // safety check, will likely never fail
                    uint64_t result_carver1 = a*x1 ^ b*z ^ structure_seed;
                    uint64_t result_carver2 = a*x2 ^ b*z ^ structure_seed;
                    if ((result_carver1 & MASK_48) == (carver1 & MASK_48) && (result_carver2 & MASK_48) == (carver2 & MASK_48)) {
                        results.push_back({structure_seed, x1, z});
                    }
                }
            }
        } 
    }

    // FIXME awful code repetition
    void reverse_given_z(uint64_t carver1, uint64_t carver2, int32_t z1, int32_t z2, std::vector<Result>& results, bool accurate_lifting = false) { 
        // zb ^ (z+d)b = C1 ^ C2 
        std::vector<uint64_t> b_values;
        if (accurate_lifting) {
            hensel_lift(0, 1, z1, z2, carver1 ^ carver2, b_values);
        }
        else {
            hensel_lift_fast(z1, z2, carver1 ^ carver2, b_values);
        }

        for (auto b : b_values) {
            uint64_t iseeds[2];
            int iseeds_count = reverse_nextLong(b & MASK_48, iseeds);

            for (int i = 0; i < iseeds_count; i++)  {
                // if iseed is for x no need to go back
                uint64_t iseed = iseeds[i];
                iseed = back2(iseed);
                uint64_t structure_seed = (iseed ^ LCG_A) & MASK_48;
                uint64_t iseed_x = iseed;

                uint64_t a = nextLong(iseed_x) & MASK_48;
                uint64_t r = (carver1 ^ b*z1 ^ structure_seed) & MASK_48;
                int tz_a = __builtin_ctzll(a);
                int tz_r = __builtin_ctzll(a);
                if (tz_a > tz_r) {
                    continue; // can't do modinv
                }
                uint64_t mod = 1ULL << 48-tz_a;
                a >>= tz_a;
                r >>= tz_a;

                uint64_t ainv = modinv(a, 48-tz_a);
                int64_t x_candidate = static_cast<int64_t>((r * ainv) & (mod-1));
                if (x_candidate >= (mod>>1)) {
                    x_candidate -= mod; // offset to reduce absolute value
                }

                if (std::abs(x_candidate) <= HALF_CHUNKS) {
                    int32_t x = static_cast<int32_t>(x_candidate);

                    // safety check, will likely never fail
                    uint64_t result_carver1 = a*x ^ b*z1 ^ structure_seed;
                    uint64_t result_carver2 = a*x ^ b*z2 ^ structure_seed;
                    if ((result_carver1 & MASK_48) == (carver1 & MASK_48) && (result_carver2 & MASK_48) == (carver2 & MASK_48)) {
                        results.push_back({structure_seed, x, z1});
                    }
                }
            }
        } 
    }

    void reverse_carver_seed_pair(uint64_t carver1, uint64_t carver2, int32_t offset_x, int32_t offset_z, std::vector<Result>& results, bool accurate_lifting = false) {
        if (offset_x != 0 && offset_z != 0) {
            printf("Error: can only specify offset on one axis.\n");
        }
        if (offset_x == 0 && offset_z == 0) {
            printf("Error: offset must be nonzero.\n");
        }
        
        if (offset_x != 0) {
            for (int32_t x = -HALF_CHUNKS; x <= HALF_CHUNKS; x++) {
                reverse_given_x(carver1, carver2, x, x+offset_x, results, accurate_lifting);
            }
        }
        else if (offset_z != 0) {
            for (int32_t z = -HALF_CHUNKS; z <= HALF_CHUNKS; z++) {
                reverse_given_z(carver1, carver2, z, z+offset_z, results, accurate_lifting);
            }
        }
    }
};

static void test_correctness() { 
    // const uint64_t a = 12845519243672438 & MASK_48; 
    // int32_t x1 = 4; 
    // int32_t x2 = 5; 
    // uint64_t xored_carvers = (x1*a ^ x2*a) & MASK_48; 
    // solve_for_a(x1, x2, xored_carvers); 

    uint64_t seed = 42792;
    int32_t z = -7;
    int32_t x = 16;
    int32_t offx[2] = {-3, 0};
    int32_t offz[2] = {0, -1};

    for (int test = 0; test < 2; test++) {
        int32_t x1 = x;
        int32_t z1 = z;
        int32_t x2 = x + offx[test];
        int32_t z2 = z + offz[test];

        uint64_t carver1 = tcr::get_carver_seed(seed, x1, z1);
        uint64_t carver2 = tcr::get_carver_seed(seed, x2, z2);

        printf("Starting reversal:  %llu ; %llu  at  %d,%d ; %d,%d\n", carver1, carver2, x1, z1, x2, z2);
        auto t0 = std::chrono::steady_clock::now();

        std::vector<tcr::Result> results;
        reverse_carver_seed_pair(carver1, carver2, offx[test], offz[test], results);

        for (auto res : results) {
            printf("Got a result! %llu : x=%d z=%d\n", res.structure_seed, res.x, res.z);
        }
        auto t1 = std::chrono::steady_clock::now();

        double ms_elapsed = (t1 - t0).count() * 1e-6;
        printf("Reversal took %f ms\n", ms_elapsed);
    }
}

int main() {
    //printf("%llu %llu\n", get_carver_seed(4382749480380LL, -494824, -1218423), get_carver_seed(4382749480380LL, -494823, -1218423));

    test_correctness();
    return 0;

    printf("Demo:\n");

    for (uint64_t c1 = 1; c1 <= 9; c1++) {
        printf("-- carver1 = %llu\n", c1);
        for (uint64_t c2 = 1; c2 <= 9; c2++) {
            std::vector<tcr::Result> results;
            tcr::reverse_carver_seed_pair(c1, c2, 1, 0, results);
            for (auto res : results) {
                printf("Got a result! %llu : x1=%d z1=%d ; x2=%d z2=%d\n", res.structure_seed, res.x, res.z, res.x+1, res.z);
            }
        }
    }
}