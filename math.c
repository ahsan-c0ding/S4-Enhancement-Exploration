#include "math.h"
#include <stdint.h>

static float bits2f(uint32_t b){ float f; memcpy(&f, &b, 4); return f; }
static uint32_t f2bits(float f){ uint32_t b; memcpy(&b, &f, 4); return b; }

/* -----------------------------------------------------------------------
 * my_exp(x) = e^x
 * x = n*ln2 + r, r in [-ln2/2, ln2/2]  =>  e^x = 2^n * e^r
 * ln2 split hi/lo (Cody-Waite) so n*ln2_hi is exact in f32 across the
 * whole supported domain. e^r: Remez-fit minimax quintic (6 coefficients).
 * 2^n: written directly into an f32's exponent bits.
 * ----------------------------------------------------------------------- */
float my_exp(float x){
    if (x < -88.0f) return 0.0f;
    if (x >  88.0f) return bits2f(0x7F800000);  /* +inf */

    const float inv_ln2 = bits2f(0x3FB8AA3B);
    const float ln2_hi  = bits2f(0x3F317200);
    const float ln2_lo  = bits2f(0x35BFBE8E);

    float t  = x * inv_ln2;
    int32_t n = (int32_t)(t >= 0.0f ? t + 0.5f : t - 0.5f);  /* round to nearest */
    float nf = (float)n;

    float r = x - nf * ln2_hi;                  // FMA
    r = r - nf * ln2_lo;                          // FMA

    float p = bits2f(0x3AB6ECC1);                 /* c6 */
    p = p * r + bits2f(0x3C0937D6);                 // FMA (c5)
    p = p * r + bits2f(0x3D2AAA0E);                 // FMA (c4)
    p = p * r + bits2f(0x3E2AAA02);                 // FMA (c3)
    p = p * r + bits2f(0x3F000000);                 // FMA (c2)
    p = p * r + bits2f(0x3F800000);                 // FMA (c1)
    p = p * r + bits2f(0x3F800000);                 // FMA (c0)

    uint32_t bits = (uint32_t)(n + 127) << 23;
    return p * bits2f(bits);
}

/* -----------------------------------------------------------------------
 * my_log(x) = ln(x).  x = m*2^e, ln(x) = e*ln2 + ln(m).
 * ln(m) via atanh double-angle identity: u = f/(f+2), f = m-1,
 *   ln(1+f) = 2*atanh(u) = 2u*(1 + u^2/3 + u^4/5 + u^6/7 + u^8/9 + u^10/11)
 * m in [1,2) => f in [0,1), but u is only in [0, 1/3) -- converges far
 * faster than a direct series in f, with no iteration/convergence risk.
 * ----------------------------------------------------------------------- */
float my_log(float x){
    if (x <= 0.0f) return -1e10f;  /* matches the original's sentinel */

    uint32_t bits = f2bits(x);
    int32_t e = (int32_t)((bits >> 23) & 0xFF) - 127;
    uint32_t mbits = (bits & 0x007FFFFF) | 0x3F800000;
    float m = bits2f(mbits);
    float f = m - 1.0f;

    float u  = f / (f + 2.0f);
    float u2 = u * u;

    float p = bits2f(0x3DA2E8BB);                  /* 1/11 */
    p = p * u2 + bits2f(0x3DE38E39);                 // FMA (1/9)
    p = p * u2 + bits2f(0x3E124925);                 // FMA (1/7)
    p = p * u2 + bits2f(0x3E4CCCCD);                 // FMA (1/5)
    p = p * u2 + bits2f(0x3EAAAAAB);                 // FMA (1/3)
    p = p * u2 + 1.0f;                                // FMA (1)

    float lnm = (2.0f * u) * p;

    const float ln2_hi = bits2f(0x3F317200);
    const float ln2_lo = bits2f(0x35BFBE8E);
    float ef = (float)e;
    float r = ef * ln2_hi + lnm;                       // FMA
    r = ef * ln2_lo + r;                                 // FMA
    return r;
}

/* -----------------------------------------------------------------------
 * my_sin(x), my_cos(x)
 * Reduce to r in [-pi,pi] via n = round(x/2pi), r = x - n*2pi, using a
 * Cody-Waite hi/lo split of 2pi. Rounding to nearest lands r in [-pi,pi]
 * directly -- no quadrant-correction branch needed (unlike the original's
 * truncate-then-correct approach).
 * sin/cos of r: fixed degree-13/12 Remez minimax polynomial in r^2 (7
 * terms) -- no loop, no division, same instruction count every call.
 * ----------------------------------------------------------------------- */
float my_sin(float x){
    const float twopi_hi  = bits2f(0x40C90F80);
    const float twopi_lo  = bits2f(0x38354443);
    const float recip_2pi = bits2f(0x3E22F983);

    float t = x * recip_2pi;
    int32_t n = (int32_t)(t >= 0.0f ? t + 0.5f : t - 0.5f);
    float nf = (float)n;

    float r = x - nf * twopi_hi;                     // FMA
    r = r - nf * twopi_lo;                             // FMA

    float x2 = r * r;
    float p = bits2f(0x2F15B1B2);                       /* c6 */
    p = p * x2 + bits2f(0xB2D46C36);                      // FMA (c5)
    p = p * x2 + bits2f(0x3638CA51);                      // FMA (c4)
    p = p * x2 + bits2f(0xB9500B09);                      // FMA (c3)
    p = p * x2 + bits2f(0x3C08887C);                      // FMA (c2)
    p = p * x2 + bits2f(0xBE2AAAAA);                      // FMA (c1)
    p = p * x2 + bits2f(0x3F800000);                      // FMA (c0)

    return r * p;
}

float my_cos(float x){
    const float twopi_hi  = bits2f(0x40C90F80);
    const float twopi_lo  = bits2f(0x38354443);
    const float recip_2pi = bits2f(0x3E22F983);

    float t = x * recip_2pi;
    int32_t n = (int32_t)(t >= 0.0f ? t + 0.5f : t - 0.5f);
    float nf = (float)n;

    float r = x - nf * twopi_hi;                      // FMA
    r = r - nf * twopi_lo;                               // FMA

    float x2 = r * r;
    /* v2: degree-14 (8-coefficient) Remez minimax, one degree higher than
     * the original optimized cut's degree-12 fit -- see "v2 CHANGELOG" at
     * the top of this file. */
    float p = bits2f(0xAD2B0E51);                          /* c7 */
    p = p * x2 + bits2f(0x310D96C7);                         // FMA (c6)
    p = p * x2 + bits2f(0xB493D39B);                         // FMA (c5)
    p = p * x2 + bits2f(0x37D00ACA);                         // FMA (c4)
    p = p * x2 + bits2f(0xBAB60B4B);                         // FMA (c3)
    p = p * x2 + bits2f(0x3D2AAAAA);                         // FMA (c2)
    p = p * x2 + bits2f(0xBF000000);                         // FMA (c1)
    p = p * x2 + bits2f(0x3F800000);                         // FMA (c0)

    return p;
}

/* -----------------------------------------------------------------------
 * my_tanh(x) = 1 - 2/(e^(2x)+1)  -- same identity as the original, built
 * on the fast my_exp() above. No clamp needed: my_exp saturates to 0 / inf
 * at the extremes, which makes this formula saturate to -1 / +1 correctly.
 * ----------------------------------------------------------------------- */
float my_tanh(float x){
    float e2x = my_exp(x + x);
    return 1.0f - 2.0f / (e2x + 1.0f);
}

/* -----------------------------------------------------------------------
 * my_pow_int(x, n) -- exponentiation by squaring: O(log n) multiplies
 * instead of the original's O(n) loop (e.g. n=64: 6 multiplies vs 64).
 * ----------------------------------------------------------------------- */
float my_pow_int(float x, int n){
    if (n < 0){
        x = 1.0f / x;
        n = -n;
    }
    float result = 1.0f;
    while (n > 0){
        if (n & 1) result *= x;
        x *= x;
        n >>= 1;
    }
    return result;
}

/* -----------------------------------------------------------------------
 * my_pow(x, y) = e^(y * ln(x)) -- structurally identical to the original,
 * automatically inherits the accuracy/speed of my_log/my_exp above.
 * ----------------------------------------------------------------------- */
float my_pow(float x, float y){
    if (x <= 0.0f) return 0.0f;
    return my_exp(y * my_log(x));
}

/* -----------------------------------------------------------------------
 * my_sqrt(x): bit-hack initial guess for 1/sqrt(x) ("fast inverse square
 * root"), refined by three Newton iterations (multiply-only, no
 * division), then sqrt(x) = x * (1/sqrt(x)).
 * This is only ever called twice per inference (a constant inside gelu()),
 * so it isn't performance-critical -- three iterations were chosen
 * (instead of two) purely to match the original's ~1e-7 accuracy rather
 * than to save instructions.
 * ----------------------------------------------------------------------- */
float my_sqrt(float x){
    if (x <= 0.0f) return 0.0f;

    float xhalf = 0.5f * x;
    uint32_t i = f2bits(x);
    i = 0x5f3759df - (i >> 1);
    float y = bits2f(i);

    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);

    return x * y;
}