#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
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
    //
    // allowMask: id 로 인덱싱되는 허용 마스크(0=제외). nullptr 이면 전체 허용.
    // id >= maskLen 인 벡터는 제외로 취급한다. 마스크는 JNI 경계를 배열 1회로
    // 건너오므로 후보별 콜백 업콜이 없다.
    // - BruteForceIndex: 스캔 루프 안에서 걸러 정확도 손실 0 (필터드 서치에 권장)
    // - HnswIndex: 그래프 탐색은 그대로 두고 결과 수집만 거른다 — 필터가 좁을수록
    //   결과가 k 미만이 될 수 있다(필터드 ANN 은 알려진 난제. 좁은 필터는 브루트포스 사용)
    virtual std::vector<SearchHit> topK(const float* query, int32_t k, const uint8_t* allowMask,
                                        int32_t maskLen) const = 0;

    std::vector<SearchHit> topK(const float* query, int32_t k) const {
        return topK(query, k, nullptr, 0);
    }

    // id 목록으로 벡터 삭제. 지원하지 않는 백엔드(HNSW)는 supportsRemove()==false 이고
    // removeByIds 가 0을 반환한다. 반환값: 실제 삭제된 개수.
    virtual bool supportsRemove() const { return false; }
    virtual size_t removeByIds(const int32_t* removeIds, size_t count) {
        (void)removeIds;
        (void)count;
        return 0;
    }

    virtual void clear() = 0;

    // ---- 직렬화 (index_io.h 의 saveIndex/loadIndex 가 호출 — 직접 사용은 내부용) ----
    virtual uint32_t formatKind() const = 0;                  // 파일 헤더의 kind 값
    virtual bool writeBody(std::FILE* f) const = 0;           // 공통 헤더 이후 본문 기록
    virtual bool readBody(std::FILE* f, int64_t count) = 0;   // 빈 인스턴스에 본문 로드
};

// Phase 0: 브루트포스(전수 비교) 구현.
class BruteForceIndex final : public VectorIndex {
public:
    static constexpr uint32_t kFormatKind = 1;

    explicit BruteForceIndex(int32_t dim);

    int32_t dim() const override { return dim_; }
    size_t size() const override { return ids_.size(); }
    void add(int32_t id, const float* vec) override;
    using VectorIndex::topK;
    std::vector<SearchHit> topK(const float* query, int32_t k, const uint8_t* allowMask,
                                int32_t maskLen) const override;
    // 연속 메모리풀 swap-remove — O(삭제수 × dim). 정확도 영향 0 (브루트포스의 강점).
    bool supportsRemove() const override { return true; }
    size_t removeByIds(const int32_t* removeIds, size_t count) override;
    void clear() override;

    uint32_t formatKind() const override { return kFormatKind; }
    bool writeBody(std::FILE* f) const override;  // ids | data
    bool readBody(std::FILE* f, int64_t count) override;

private:
    int32_t dim_;
    std::vector<int32_t> ids_;
    std::vector<float> data_;  // row-major (ids_.size() x dim_), L2 정규화 저장
};

}  // namespace rag
