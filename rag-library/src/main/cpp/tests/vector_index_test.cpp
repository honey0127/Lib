// rag::VectorIndex 를 Android 없이 호스트에서 검증하는 독립 실행 테스트.
// (CMake 타깃에 포함되지 않는다 — CI/개발 PC에서 아래 명령으로 직접 실행)
//
//   g++ -std=c++17 -O2 rag-library/src/main/cpp/vector_index.cpp
//       rag-library/src/main/cpp/tests/vector_index_test.cpp
//       -o /tmp/vector_index_test  (한 줄로 이어서 실행)

#include "../vector_index.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../dot_product.h"

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
    using VectorIndex = rag::BruteForceIndex;

    // 1) 정규화 + 코사인 값 + 정렬: 정규화 안 된 입력도 동일 결과여야 한다
    {
        VectorIndex idx(3);
        float a[3] = {2.0f, 0.0f, 0.0f};  // → (1,0,0)
        float b[3] = {0.0f, 5.0f, 0.0f};  // → (0,1,0)
        float c[3] = {3.0f, 3.0f, 0.0f};  // → (0.7071, 0.7071, 0)
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

    // 2) k 가 크기보다 크면 클램프, k <= 0 이면 빈 결과
    {
        VectorIndex idx(2);
        float e0[2] = {1.0f, 0.0f};
        idx.add(0, e0);
        expect(idx.topK(e0, 5).size() == 1, "k > size clamps to size");
        expect(idx.topK(e0, 0).empty(), "k == 0 -> empty");
        expect(idx.topK(e0, -3).empty(), "k < 0 -> empty");
    }

    // 3) 반대 방향 벡터 → 코사인 -1
    {
        VectorIndex idx(2);
        float pos[2] = {1.0f, 0.0f};
        float neg[2] = {-4.0f, 0.0f};
        idx.add(1, pos);
        idx.add(2, neg);
        auto hits = idx.topK(pos, 2);
        expect(hits[0].id == 1 && near(hits[0].score, 1.0f), "same dir -> 1.0");
        expect(hits[1].id == 2 && near(hits[1].score, -1.0f), "opposite dir -> -1.0");
    }

    // 4) 영벡터 add/query: NaN 없이 점수 0
    {
        VectorIndex idx(2);
        float zero[2] = {0.0f, 0.0f};
        float e0[2] = {1.0f, 0.0f};
        idx.add(1, zero);
        idx.add(2, e0);
        auto hits = idx.topK(e0, 2);
        expect(hits[0].id == 2 && near(hits[0].score, 1.0f), "unit vec ranks first");
        expect(hits[1].id == 1 && near(hits[1].score, 0.0f), "zero vec scores 0");
        for (auto& h : idx.topK(zero, 2)) {
            expect(!std::isnan(h.score) && near(h.score, 0.0f), "zero query -> all scores 0, no NaN");
        }
    }

    // 5) clear 후 빈 결과
    {
        VectorIndex idx(2);
        float e0[2] = {1.0f, 0.0f};
        idx.add(1, e0);
        idx.clear();
        expect(idx.size() == 0, "clear -> size 0");
        expect(idx.topK(e0, 3).empty(), "clear -> empty topK");
    }

    // 5.1) allowMask 필터: 스캔 루프 내 정확 필터링
    {
        VectorIndex idx(2);
        float a[2] = {1.0f, 0.0f};   // id 0 — 질의와 동일
        float b[2] = {0.9f, 0.1f};   // id 1 — 근접
        float c[2] = {0.0f, 1.0f};   // id 2 — 직교
        idx.add(0, a);
        idx.add(1, b);
        idx.add(2, c);
        const uint8_t mask[3] = {0, 1, 1};  // id 0 제외
        auto hits = idx.topK(a, 3, mask, 3);
        expect(hits.size() == 2, "filter excludes masked-out id");
        expect(hits[0].id == 1 && hits[1].id == 2, "filtered order correct");
        const uint8_t none[3] = {0, 0, 0};
        expect(idx.topK(a, 3, none, 3).empty(), "all-masked -> empty");
        // 마스크 범위 밖 id 는 제외 취급
        expect(idx.topK(a, 3, mask, 1).empty(), "ids >= maskLen excluded");
    }

    // 5.2) removeByIds: swap-remove 후 검색/추가 정상
    {
        VectorIndex idx(2);
        float a[2] = {1.0f, 0.0f};
        float b[2] = {0.0f, 1.0f};
        float c[2] = {-1.0f, 0.0f};
        idx.add(10, a);
        idx.add(11, b);
        idx.add(12, c);
        expect(idx.supportsRemove(), "brute supports remove");
        const int32_t rm[2] = {11, 99};  // 99 는 없음
        expect(idx.removeByIds(rm, 2) == 1, "remove returns actual count");
        expect(idx.size() == 2, "size after remove");
        auto hits = idx.topK(a, 3);
        expect(hits.size() == 2 && hits[0].id == 10 && hits[1].id == 12,
               "removed id absent, others intact");
        idx.add(13, b);  // 삭제 후 재추가
        expect(idx.topK(b, 1).front().id == 13, "add after remove works");
    }

    // 5.5) SIMD dotProduct == 스칼라 참조 (홀수 길이 꼬리 포함, 허용 오차 내)
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
        for (int32_t dim = 1; dim <= 80; ++dim) {
            std::vector<float> a(static_cast<size_t>(dim));
            std::vector<float> b(static_cast<size_t>(dim));
            for (auto& x : a) x = ud(gen);
            for (auto& x : b) x = ud(gen);
            float ref = 0.0f;
            for (int32_t j = 0; j < dim; ++j) ref += a[static_cast<size_t>(j)] * b[static_cast<size_t>(j)];
            const float got = rag::dotProduct(a.data(), b.data(), dim);
            if (std::fabs(got - ref) > 1e-4f) {
                std::printf("FAIL: dotProduct dim=%d got=%f ref=%f\n", dim, got, ref);
                ++g_failures;
            }
        }
    }

    // 6) 큰 차원(임베딩 크기 수준)에서 자기 자신 검색
    {
        const int32_t dim = 384;
        VectorIndex idx(dim);
        std::vector<float> v(dim);
        for (int32_t i = 0; i < dim; ++i) v[i] = static_cast<float>(std::rand() % 100 - 50) / 10.0f;
        idx.add(7, v.data());
        auto hits = idx.topK(v.data(), 1);
        expect(hits.size() == 1 && hits[0].id == 7 && near(hits[0].score, 1.0f, 1e-4f),
               "self-search on dim=384 -> 1.0");
    }

    if (g_failures == 0) {
        std::printf("vector_index_test: ALL OK\n");
        return 0;
    }
    std::printf("vector_index_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
