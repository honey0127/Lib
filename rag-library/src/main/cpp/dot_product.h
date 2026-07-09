#pragma once

#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define RAG_DOT_NEON 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define RAG_DOT_SSE2 1
#endif

namespace rag {

// 내적 — 인덱스 검색의 최심부 핫루프. ABI별 SIMD 경로:
//   arm64-v8a → NEON(vfmaq, 8-wide 언롤), x86_64/호스트 → SSE2(8-wide 언롤),
//   그 외 → 4-누산기 스칼라(ILP).
// SIMD 는 합산 순서가 스칼라와 달라 결과가 최하위 비트 수준에서 다를 수 있다
// (코사인 순위에는 영향 없음 — tests/vector_index_test.cpp 에서 스칼라 대비 오차 검증).
inline float dotProduct(const float* a, const float* b, int32_t n) {
#if defined(RAG_DOT_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    for (; i + 4 <= n; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#elif defined(RAG_DOT_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    __m128 acc = _mm_add_ps(acc0, acc1);
    __m128 shuf = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1));
    acc = _mm_add_ps(acc, shuf);
    shuf = _mm_movehl_ps(shuf, acc);
    acc = _mm_add_ss(acc, shuf);
    float sum = _mm_cvtss_f32(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

}  // namespace rag
