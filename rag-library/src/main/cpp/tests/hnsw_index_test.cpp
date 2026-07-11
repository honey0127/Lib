// rag::HnswIndex 호스트 단독 테스트 — 정확성(소규모) + 재현율(브루트포스 대비).
// (CMake 타깃에 포함되지 않는다 — CI/개발 PC에서 아래 명령으로 직접 실행)
//
//   g++ -std=c++17 -O2 rag-library/src/main/cpp/vector_index.cpp
//       rag-library/src/main/cpp/hnsw_index.cpp
//       rag-library/src/main/cpp/tests/hnsw_index_test.cpp
//       -o /tmp/hnsw_index_test  (한 줄로 이어서 실행)

#include "../hnsw_index.h"
#include "../vector_index.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool near(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

}  // namespace

int main() {
    // 1) 소규모 정확성: 기본 ef(64) >> n(3) → 사실상 정확 탐색, 브루트포스와 동일해야 함
    {
        rag::HnswIndex idx(3);
        float a[3] = {2.0f, 0.0f, 0.0f};
        float b[3] = {0.0f, 5.0f, 0.0f};
        float c[3] = {3.0f, 3.0f, 0.0f};
        idx.add(10, a);
        idx.add(11, b);
        idx.add(12, c);
        expect(idx.size() == 3, "size == 3");

        float q[3] = {7.0f, 0.0f, 0.0f};
        auto hits = idx.topK(q, 3);
        expect(hits.size() == 3, "topK(3) returns 3");
        expect(hits[0].id == 10 && near(hits[0].score, 1.0f), "1st: id=10 score=1.0");
        expect(hits[1].id == 12 && near(hits[1].score, 0.70710678f), "2nd: id=12 score=0.7071");
        expect(hits[2].id == 11 && near(hits[2].score, 0.0f), "3rd: id=11 score=0.0");
    }

    // 2) 경계: k 클램프 / k<=0 / 빈 인덱스 / clear
    {
        rag::HnswIndex idx(2);
        float e0[2] = {1.0f, 0.0f};
        expect(idx.topK(e0, 5).empty(), "empty index -> empty");
        idx.add(0, e0);
        expect(idx.topK(e0, 5).size() == 1, "k > size clamps to size");
        expect(idx.topK(e0, 0).empty(), "k == 0 -> empty");
        expect(idx.topK(e0, -1).empty(), "k < 0 -> empty");
        idx.clear();
        expect(idx.size() == 0 && idx.topK(e0, 3).empty(), "clear -> empty");
    }

    // 3) 자기 자신 검색: 200개 랜덤 벡터에서 각자 top-1 은 자기 자신
    {
        const int32_t dim = 16;
        rag::HnswIndex idx(dim);
        std::mt19937 gen(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<std::vector<float>> vecs(200, std::vector<float>(dim));
        for (int32_t i = 0; i < 200; ++i) {
            for (auto& x : vecs[static_cast<size_t>(i)]) x = nd(gen);
            idx.add(i, vecs[static_cast<size_t>(i)].data());
        }
        int selfHits = 0;
        for (int32_t i = 0; i < 200; ++i) {
            auto hits = idx.topK(vecs[static_cast<size_t>(i)].data(), 1);
            if (hits.size() == 1 && hits[0].id == i && near(hits[0].score, 1.0f, 1e-4f)) {
                ++selfHits;
            }
        }
        std::printf("self-search: %d/200\n", selfHits);
        expect(selfHits >= 198, "self-search >= 198/200");
    }

    // 4) 재현율: 브루트포스 대비 recall@10 (n=2000, dim=32, 질의 50개)
    {
        const int32_t dim = 32, N = 2000, Q = 50, K = 10;
        rag::BruteForceIndex exact(dim);
        rag::HnswIndex ann(dim, /*m=*/16, /*efConstruction=*/200, /*efSearch=*/100);
        std::mt19937 gen(123);
        std::normal_distribution<float> nd(0.0f, 1.0f);

        std::vector<float> v(static_cast<size_t>(dim));
        for (int32_t i = 0; i < N; ++i) {
            for (auto& x : v) x = nd(gen);
            exact.add(i, v.data());
            ann.add(i, v.data());
        }

        int match = 0;
        for (int32_t qi = 0; qi < Q; ++qi) {
            for (auto& x : v) x = nd(gen);
            auto gt = exact.topK(v.data(), K);
            auto ae = ann.topK(v.data(), K);
            expect(ae.size() == static_cast<size_t>(K), "hnsw returns K results");
            std::set<int32_t> gtIds;
            for (const auto& h : gt) gtIds.insert(h.id);
            for (const auto& h : ae) {
                if (gtIds.count(h.id)) ++match;
            }
        }
        const double recall = static_cast<double>(match) / (Q * K);
        std::printf("HNSW recall@%d = %.3f (n=%d, efSearch=100)\n", K, recall, N);
        expect(recall >= 0.90, "recall@10 >= 0.90");
    }

    // 4.5) allowMask 포스트 필터: 허용된 id 만 반환
    {
        rag::HnswIndex idx(3);
        float a[3] = {1.0f, 0.0f, 0.0f};
        float b[3] = {0.9f, 0.1f, 0.0f};
        float c[3] = {0.0f, 1.0f, 0.0f};
        idx.add(0, a);
        idx.add(1, b);
        idx.add(2, c);
        const uint8_t mask[3] = {0, 1, 1};
        auto hits = idx.topK(a, 3, mask, 3);
        expect(hits.size() == 2 && hits[0].id == 1 && hits[1].id == 2,
               "hnsw filter returns only allowed ids");
        expect(!idx.supportsRemove(), "hnsw does not support remove");
    }

    // 5) 영벡터 add/query — NaN 없이 동작
    {
        rag::HnswIndex idx(2);
        float zero[2] = {0.0f, 0.0f};
        float e0[2] = {1.0f, 0.0f};
        idx.add(1, zero);
        idx.add(2, e0);
        auto hits = idx.topK(e0, 2);
        expect(hits.size() == 2, "zero-vec index returns 2");
        expect(hits[0].id == 2 && near(hits[0].score, 1.0f), "unit vec first");
        for (const auto& h : idx.topK(zero, 2)) {
            expect(!std::isnan(h.score) && near(h.score, 0.0f), "zero query -> score 0, no NaN");
        }
    }

    if (g_failures == 0) {
        std::printf("hnsw_index_test: ALL OK\n");
        return 0;
    }
    std::printf("hnsw_index_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
