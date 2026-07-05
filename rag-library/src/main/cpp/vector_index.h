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

// Phase 0: 브루트포스 코사인 유사도 인덱스.
// - 벡터는 add() 시점에 L2 정규화되어 저장되므로 topK()의 내적 = 코사인 유사도.
// - 스레드 안전하지 않다. 동기화는 호출측(Kotlin RagEngine)이 담당.
// - Phase 1 에서 동일 인터페이스를 유지한 채 HNSW 로 교체할 예정.
class VectorIndex {
public:
    explicit VectorIndex(int32_t dim);

    int32_t dim() const { return dim_; }
    size_t size() const { return ids_.size(); }

    // vec 은 dim() 길이의 배열이어야 한다(호출측 JNI 레이어에서 검증).
    // 영벡터는 정규화 없이 0 벡터로 저장된다(점수 항상 0).
    void add(int32_t id, const float* vec);

    // query 와의 코사인 유사도 상위 k개를 점수 내림차순으로 반환.
    // k 가 size() 보다 크면 size() 개, k <= 0 이면 빈 결과.
    std::vector<SearchHit> topK(const float* query, int32_t k) const;

    void clear();

private:
    int32_t dim_;
    std::vector<int32_t> ids_;
    std::vector<float> data_;  // row-major (ids_.size() x dim_), L2 정규화 저장
};

}  // namespace rag
