// 인덱스 직렬화(saveIndex/loadIndex) 호스트 단독 테스트.
// (CMake 타깃에 포함되지 않는다 — CI/개발 PC에서 아래 명령으로 직접 실행)
//
//   g++ -std=c++17 -O2 rag-library/src/main/cpp/vector_index.cpp
//       rag-library/src/main/cpp/hnsw_index.cpp
//       rag-library/src/main/cpp/index_io.cpp
//       rag-library/src/main/cpp/tests/index_io_test.cpp
//       -o /tmp/index_io_test  (한 줄로 이어서 실행)

#include "../index_io.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../hnsw_index.h"
#include "../vector_index.h"

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

bool near(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) < eps; }

// 두 인덱스가 같은 질의에 같은 결과를 내는지 (id 동일 + 점수 근사)
bool sameResults(const rag::VectorIndex& a, const rag::VectorIndex& b, const float* q, int32_t k) {
    const auto ha = a.topK(q, k);
    const auto hb = b.topK(q, k);
    if (ha.size() != hb.size()) {
        return false;
    }
    for (size_t i = 0; i < ha.size(); ++i) {
        if (ha[i].id != hb[i].id || !near(ha[i].score, hb[i].score)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const char* brutePath = "/tmp/rag_io_test_brute.bin";
    const char* hnswPath = "/tmp/rag_io_test_hnsw.bin";

    std::mt19937 gen(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int32_t dim = 16;

    // 1) 브루트포스 왕복: 저장 → 로드 → 동일 검색 결과
    {
        rag::BruteForceIndex idx(dim);
        std::vector<float> v(static_cast<size_t>(dim));
        for (int32_t i = 0; i < 100; ++i) {
            for (auto& x : v) x = nd(gen);
            idx.add(i * 7, v.data());  // id 가 노드 순번과 다르게
        }
        expect(rag::saveIndex(idx, brutePath), "brute save ok");

        auto loaded = rag::loadIndex(brutePath);
        expect(loaded != nullptr, "brute load ok");
        if (loaded) {
            expect(loaded->dim() == dim && loaded->size() == 100, "brute dim/size preserved");
            for (int32_t t = 0; t < 5; ++t) {
                for (auto& x : v) x = nd(gen);
                expect(sameResults(idx, *loaded, v.data(), 10), "brute same topK after load");
            }
            // 로드 후 추가 삽입도 정상 동작
            for (auto& x : v) x = nd(gen);
            loaded->add(9999, v.data());
            auto hits = loaded->topK(v.data(), 1);
            expect(hits.size() == 1 && hits[0].id == 9999 && near(hits[0].score, 1.0f, 1e-4f),
                   "brute add-after-load works");
        }
    }

    // 2) HNSW 왕복: 그래프까지 복원 → 결과 동일
    {
        rag::HnswIndex idx(dim, 8, 100, 50);
        std::vector<float> v(static_cast<size_t>(dim));
        for (int32_t i = 0; i < 300; ++i) {
            for (auto& x : v) x = nd(gen);
            idx.add(i, v.data());
        }
        expect(rag::saveIndex(idx, hnswPath), "hnsw save ok");

        auto loaded = rag::loadIndex(hnswPath);
        expect(loaded != nullptr, "hnsw load ok");
        if (loaded) {
            expect(loaded->dim() == dim && loaded->size() == 300, "hnsw dim/size preserved");
            for (int32_t t = 0; t < 10; ++t) {
                for (auto& x : v) x = nd(gen);
                expect(sameResults(idx, *loaded, v.data(), 10), "hnsw same topK after load");
            }
            for (auto& x : v) x = nd(gen);
            loaded->add(8888, v.data());
            auto hits = loaded->topK(v.data(), 1);
            expect(hits.size() == 1 && hits[0].id == 8888 && near(hits[0].score, 1.0f, 1e-4f),
                   "hnsw add-after-load works");
        }
    }

    // 3) 빈 인덱스 왕복
    {
        rag::HnswIndex empty(4);
        expect(rag::saveIndex(empty, hnswPath), "empty save ok");
        auto loaded = rag::loadIndex(hnswPath);
        expect(loaded != nullptr && loaded->size() == 0, "empty load ok");
        if (loaded) {
            float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
            expect(loaded->topK(q, 3).empty(), "empty loaded topK empty");
            loaded->add(1, q);
            expect(loaded->topK(q, 3).size() == 1, "empty loaded add works");
        }
    }

    // 4) 손상/비정상 파일 거부
    {
        expect(rag::loadIndex("/tmp/rag_io_test_missing.bin") == nullptr, "missing file -> null");

        std::FILE* f = std::fopen(brutePath, "wb");
        const char garbage[] = "not an index file";
        std::fwrite(garbage, 1, sizeof(garbage), f);
        std::fclose(f);
        expect(rag::loadIndex(brutePath) == nullptr, "bad magic -> null");

        // 정상 헤더 + 잘린 본문
        rag::BruteForceIndex idx(dim);
        std::vector<float> v(static_cast<size_t>(dim), 1.0f);
        for (int32_t i = 0; i < 10; ++i) idx.add(i, v.data());
        expect(rag::saveIndex(idx, brutePath), "truncation setup save ok");
        f = std::fopen(brutePath, "rb");
        std::fseek(f, 0, SEEK_END);
        const long full = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<char> buf(static_cast<size_t>(full / 2));
        expect(std::fread(buf.data(), 1, buf.size(), f) == buf.size(), "read half");
        std::fclose(f);
        f = std::fopen(brutePath, "wb");
        std::fwrite(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        expect(rag::loadIndex(brutePath) == nullptr, "truncated file -> null");
    }

    std::remove(brutePath);
    std::remove(hnswPath);

    if (g_failures == 0) {
        std::printf("index_io_test: ALL OK\n");
        return 0;
    }
    std::printf("index_io_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
