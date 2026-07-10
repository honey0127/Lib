// 검색 코어 성능 벤치마크 — 호스트에서 수동 실행 (CI 미포함, 러너 타이밍은 노이즈).
//
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror rag-library/src/main/cpp/vector_index.cpp
//       rag-library/src/main/cpp/hnsw_index.cpp rag-library/src/main/cpp/tests/bench.cpp
//       -o /tmp/rag_bench && /tmp/rag_bench   (한 줄로 이어서 실행)
//
// 시나리오: e5 임베딩 차원(384), 2만 청크, 질의 100개, top-10.

#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "../dot_product.h"
#include "../hnsw_index.h"
#include "../vector_index.h"

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// SIMD 비교용 스칼라 참조 (엄격한 FP 의미론이 자동 벡터화를 막는다)
float scalarDot(const float* a, const float* b, int32_t n) {
    float s = 0.0f;
    for (int32_t j = 0; j < n; ++j) {
        s += a[j] * b[j];
    }
    return s;
}

}  // namespace

int main() {
    const int32_t dim = 384;
    const int32_t N = 20000;
    const int32_t NMAX = 50000;
    const int32_t Q = 100;
    const int32_t K = 10;

    std::mt19937 gen(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(NMAX) * dim);
    for (auto& x : data) x = nd(gen);
    std::vector<float> queries(static_cast<size_t>(Q) * dim);
    for (auto& x : queries) x = nd(gen);

    std::printf("dim=%d N=%d Q=%d K=%d\n", dim, N, Q, K);

    // ---- 0) dot product 마이크로벤치: 스칼라 vs SIMD (동일 데이터 2천만 회) ----
    {
        const int32_t reps = 5;
        volatile float sink = 0.0f;
        auto t0 = Clock::now();
        for (int32_t r = 0; r < reps; ++r) {
            for (int32_t i = 0; i < N; ++i) {
                sink = sink + scalarDot(data.data() + static_cast<size_t>(i) * dim,
                                        queries.data(), dim);
            }
        }
        const double scalarMs = msSince(t0);
        t0 = Clock::now();
        for (int32_t r = 0; r < reps; ++r) {
            for (int32_t i = 0; i < N; ++i) {
                sink = sink + rag::dotProduct(data.data() + static_cast<size_t>(i) * dim,
                                              queries.data(), dim);
            }
        }
        const double simdMs = msSince(t0);
        std::printf("dot(%d) x %dk : 스칼라 %7.1f ms | SIMD %7.1f ms | %.1fx\n", dim,
                    N * reps / 1000, scalarMs, simdMs, scalarMs / simdMs);
    }

    // ---- 1) BruteForceIndex — 코퍼스 규모 스윕 ----
    rag::BruteForceIndex brute(dim);
    double checksum = 0.0;  // 최적화 제거 방지
    {
        int32_t added = 0;
        const auto t0 = Clock::now();
        for (const int32_t target : {1000, 10000, 20000, 50000}) {
            while (added < target) {
                brute.add(added, data.data() + static_cast<size_t>(added) * dim);
                ++added;
            }
            const double addMs = msSince(t0);
            const auto tq = Clock::now();
            for (int32_t qi = 0; qi < Q; ++qi) {
                const auto hits = brute.topK(queries.data() + static_cast<size_t>(qi) * dim, K);
                checksum += hits.front().score;
            }
            std::printf("brute  N=%5d: search %7.3f ms/질의 (누적 add %.1f ms, %.1f µs/벡터)\n",
                        target, msSince(tq) / Q, addMs, addMs * 1000.0 / target);
        }
        // 이후 HNSW 비교는 N=20000 기준 — 브루트포스를 20k 로 재구성
        brute.clear();
        for (int32_t i = 0; i < N; ++i) {
            brute.add(i, data.data() + static_cast<size_t>(i) * dim);
        }
    }

    // ---- HnswIndex ----
    rag::HnswIndex hnsw(dim, 16, 200, 100);
    {
        const auto t0 = Clock::now();
        for (int32_t i = 0; i < N; ++i) {
            hnsw.add(i, data.data() + static_cast<size_t>(i) * dim);
        }
        std::printf("hnsw   build : %8.1f ms 총계\n", msSince(t0));
    }
    {
        int match = 0;
        const auto t0 = Clock::now();
        for (int32_t qi = 0; qi < Q; ++qi) {
            const auto hits = hnsw.topK(queries.data() + static_cast<size_t>(qi) * dim, K);
            checksum += hits.front().score;
        }
        const double perQuery = msSince(t0) / Q;
        for (int32_t qi = 0; qi < Q; ++qi) {
            const float* q = queries.data() + static_cast<size_t>(qi) * dim;
            const auto gt = brute.topK(q, K);
            const auto ae = hnsw.topK(q, K);
            std::set<int32_t> ids;
            for (const auto& h : gt) ids.insert(h.id);
            for (const auto& h : ae) match += static_cast<int>(ids.count(h.id));
        }
        std::printf("hnsw   search: %8.3f ms/질의  (recall@%d = %.3f)\n", perQuery, K,
                    static_cast<double>(match) / (Q * K));
    }

    std::printf("(checksum %.4f)\n", checksum);
    return 0;
}
