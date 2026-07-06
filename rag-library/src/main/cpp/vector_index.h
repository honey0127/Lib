#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rag {

// 검색 결과 1건: 벡터 추가 시 부여한 id 와 코사인 유사도 점수(-1 ~ 1).
struct SearchHit {
    int32_t id;
    float score;
};

// v 를 제자리 L2 정규화. 노름이 0에 가까우면 0 벡터로 만든다(NaN 방지).
void l2NormalizeInPlace(float* v, int32_t dim);

// 벡터 인덱스 공통 인터페이스.
// - 벡터는 add() 시점에 L2 정규화되어 저장되므로 topK()의 내적 = 코사인 유사도.
// - 스레드 안전하지 않다. 동기화는 호출측(Kotlin RagEngine)이 담당.
// 구현체:
// - BruteForceIndex : 정확(exact) 전수 비교 — 수천 청크까지는 빠르고 항상 정확 (기본값)
// - HnswIndex       : 근사(ANN) 그래프 탐색 — 대규모 코퍼스용 (hnsw_index.h)
class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    virtual int32_t dim() const = 0;
    virtual size_t size() const = 0;

    // vec 은 dim() 길이의 배열이어야 한다(호출측 JNI 레이어에서 검증).
    // 영벡터는 정규화 없이 0 벡터로 저장된다(점수 항상 0).
    virtual void add(int32_t id, const float* vec) = 0;

    // query 와의 코사인 유사도 상위 k개를 점수 내림차순으로 반환.
    // k 가 size() 보다 크면 size() 개, k <= 0 이면 빈 결과.
    virtual std::vector<SearchHit> topK(const float* query, int32_t k) const = 0;

    virtual void clear() = 0;
};

// Phase 0: 브루트포스(전수 비교) 구현.
class BruteForceIndex final : public VectorIndex {
public:
    explicit BruteForceIndex(int32_t dim);

    int32_t dim() const override { return dim_; }
    size_t size() const override { return ids_.size(); }
    void add(int32_t id, const float* vec) override;
    std::vector<SearchHit> topK(const float* query, int32_t k) const override;
    void clear() override;

private:
    int32_t dim_;
    std::vector<int32_t> ids_;
    std::vector<float> data_;  // row-major (ids_.size() x dim_), L2 정규화 저장
};

}  // namespace rag
