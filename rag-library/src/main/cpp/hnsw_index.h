#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "vector_index.h"

namespace rag {

// HNSW(Hierarchical Navigable Small World) 근사 최근접 이웃 인덱스.
// Malkov & Yashunin (2016) — 다층 그래프에서 greedy 하강 + 레벨 0 beam 탐색.
//
// - 거리: 코사인 거리(1 - dot). 벡터는 add() 시 L2 정규화 저장.
// - Phase 1: BruteForceIndex 와 동일한 VectorIndex 인터페이스 — 교체 가능.
// - 파라미터: m(레벨당 연결 수), efConstruction(삽입 시 beam 폭),
//   efSearch(검색 시 beam 폭, 클수록 재현율↑ 속도↓).
// - 결정성: 레벨 추첨이 내부 시드 고정 rng 를 쓰므로 같은 삽입 순서 → 같은 그래프.
class HnswIndex final : public VectorIndex {
public:
    static constexpr uint32_t kFormatKind = 2;

    explicit HnswIndex(int32_t dim, int32_t m = 16, int32_t efConstruction = 200,
                       int32_t efSearch = 128, uint64_t seed = 42);

    int32_t dim() const override { return dim_; }
    size_t size() const override { return ids_.size(); }
    void add(int32_t id, const float* vec) override;
    using VectorIndex::topK;
    // allowMask: 탐색은 전체 그래프에서 하되 결과 수집만 거른다(포스트 필터).
    // 필터가 좁으면 k 미만이 반환될 수 있다 — 좁은 필터는 BruteForceIndex 권장.
    std::vector<SearchHit> topK(const float* query, int32_t k, const uint8_t* allowMask,
                                int32_t maskLen) const override;
    // 그래프 노드 삭제는 미지원(연결성 훼손 — 알려진 난제). supportsRemove()==false.
    void clear() override;

    uint32_t formatKind() const override { return kFormatKind; }
    // 본문: m | efC | efS | entry | maxLevel | ids | data | 노드별(레벨수, 레벨별 이웃수+이웃)
    bool writeBody(std::FILE* f) const override;
    // 로드 후 그래프는 저장 시점과 동일 → 검색 결과 동일. 이후 add() 는 새 rng 시퀀스 사용
    // (그래프 품질 동일, 시드 기반 결정성만 저장 전과 달라짐).
    bool readBody(std::FILE* f, int64_t count) override;

private:
    struct Cand {
        float dist;
        int32_t node;
    };

    const float* row(int32_t node) const {
        return data_.data() + static_cast<size_t>(node) * dim_;
    }
    float distTo(const float* q, int32_t node) const;
    int32_t randomLevel();
    // level 그래프에서 start 로부터 greedy 로 가장 가까운 노드로 이동
    int32_t greedyClosest(const float* q, int32_t start, int32_t level) const;
    // level 그래프에서 entry 기준 ef 개 근접 후보를 dist 오름차순으로 반환
    std::vector<Cand> searchLayer(const float* q, int32_t entry, int32_t level, int32_t ef) const;
    // Malkov Algorithm 4 — 다양성 휴리스틱 이웃 선택 (cands 는 dist 오름차순).
    // 후보가 이미 뽑힌 이웃보다 질의점에 더 가까울 때만 채택해 방향 다양성을 확보한다.
    // (고차원·대규모에서 단순 최근접 M개 선택은 재현율을 무너뜨린다 — bench.cpp 로 검증)
    std::vector<Cand> selectNeighbors(const std::vector<Cand>& cands, int32_t m) const;
    // node 와 선택된 이웃을 양방향 연결하고, 초과 연결은 휴리스틱으로 재선택
    void linkNeighbors(int32_t node, int32_t level, const std::vector<Cand>& cands);

    int32_t dim_;
    int32_t m_;                // 상위 레벨 최대 연결 수
    int32_t maxM0_;            // 레벨 0 최대 연결 수 (= 2 * m_)
    int32_t efConstruction_;
    int32_t efSearch_;
    double levelMult_;         // 1 / ln(m)
    std::mt19937_64 rng_;

    std::vector<int32_t> ids_;
    std::vector<float> data_;                               // 정규화 저장, row-major
    std::vector<std::vector<std::vector<int32_t>>> links_;  // links_[node][level] = 이웃 목록
    int32_t entry_ = -1;
    int32_t maxLevel_ = -1;

    // 탐색용 방문 표시(스탬프 방식) — topK(const)에서도 쓰므로 mutable
    mutable std::vector<uint32_t> visitedStamp_;
    mutable uint32_t stamp_ = 0;
};

}  // namespace rag
