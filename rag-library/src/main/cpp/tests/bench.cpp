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

#include "../hnsw_index.h"
#include "../vector_index.h"

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main() {
    const int32_t dim = 384;
    const int32_t N = 20000;
    const int32_t Q = 100;
    const int32_t K = 10;

    std::mt19937 gen(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(N) * dim);
    for (auto& x : data) x = nd(gen);
    std::vector<float> queries(static_cast<size_t>(Q) * dim);
    for (auto& x : queries) x = nd(gen);

    std::printf("dim=%d N=%d Q=%d K=%d\n", dim, N, Q, K);

    // ---- BruteForceIndex ----
    rag::BruteForceIndex brute(dim);
    {
        const auto t0 = Clock::now();
        for (int32_t i = 0; i < N; ++i) {
            brute.add(i, data.data() + static_cast<size_t>(i) * dim);
        }
        std::printf("brute  add   : %8.1f ms 총계\n", msSince(t0));
    }
    double checksum = 0.0;  // 최적화 제거 방지
    {
        const auto t0 = Clock::now();
        for (int32_t qi = 0; qi < Q; ++qi) {
            const auto hits = brute.topK(queries.data() + static_cast<size_t>(qi) * dim, K);
            checksum += hits.front().score;
        }
        std::printf("brute  search: %8.3f ms/질의\n", msSince(t0) / Q);
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
