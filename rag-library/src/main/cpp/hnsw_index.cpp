#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <queue>

#include "index_io.h"

namespace rag {

namespace {
// 파일 로드 시 허용하는 최대 원소 수 (손상 파일의 과대 할당 방지, ~1GB)
constexpr size_t kMaxLoadFloats = size_t(1) << 28;
constexpr int32_t kMaxLevels = 64;
}

HnswIndex::HnswIndex(int32_t dim, int32_t m, int32_t efConstruction, int32_t efSearch,
                     uint64_t seed)
    : dim_(dim),
      m_(m < 2 ? 2 : m),
      maxM0_(2 * (m < 2 ? 2 : m)),
      efConstruction_(efConstruction < m_ ? m_ : efConstruction),
      efSearch_(efSearch < 1 ? 1 : efSearch),
      levelMult_(1.0 / std::log(static_cast<double>(m_ < 2 ? 2 : m_))),
      rng_(seed) {}

float HnswIndex::distTo(const float* q, int32_t node) const {
    const float* r = row(node);
    float dot = 0.0f;
    for (int32_t j = 0; j < dim_; ++j) {
        dot += r[j] * q[j];
    }
    return 1.0f - dot;
}

int32_t HnswIndex::randomLevel() {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double x = uniform(rng_);
    if (x < 1e-12) {
        x = 1e-12;
    }
    return static_cast<int32_t>(-std::log(x) * levelMult_);
}

int32_t HnswIndex::greedyClosest(const float* q, int32_t start, int32_t level) const {
    int32_t cur = start;
    float curDist = distTo(q, cur);
    bool improved = true;
    while (improved) {
        improved = false;
        for (const int32_t nb : links_[cur][level]) {
            const float d = distTo(q, nb);
            if (d < curDist) {
                curDist = d;
                cur = nb;
                improved = true;
            }
        }
    }
    return cur;
}

std::vector<HnswIndex::Cand> HnswIndex::searchLayer(const float* q, int32_t entry, int32_t level,
                                                    int32_t ef) const {
    ++stamp_;
    if (stamp_ == 0) {  // uint32 랩어라운드 — 사실상 도달 불가지만 안전하게
        std::fill(visitedStamp_.begin(), visitedStamp_.end(), 0u);
        stamp_ = 1;
    }
    if (visitedStamp_.size() < ids_.size()) {
        visitedStamp_.resize(ids_.size(), 0u);
    }

    const auto nearerOnTop = [](const Cand& a, const Cand& b) { return a.dist > b.dist; };
    const auto fartherOnTop = [](const Cand& a, const Cand& b) { return a.dist < b.dist; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(nearerOnTop)> candidates(nearerOnTop);
    std::priority_queue<Cand, std::vector<Cand>, decltype(fartherOnTop)> result(fartherOnTop);

    const float d0 = distTo(q, entry);
    candidates.push({d0, entry});
    result.push({d0, entry});
    visitedStamp_[static_cast<size_t>(entry)] = stamp_;

    while (!candidates.empty()) {
        const Cand c = candidates.top();
        if (static_cast<int32_t>(result.size()) >= ef && c.dist > result.top().dist) {
            break;  // 가장 가까운 미확장 후보가 결과 최악보다 멀면 종료
        }
        candidates.pop();

        const auto& nodeLinks = links_[static_cast<size_t>(c.node)];
        if (level >= static_cast<int32_t>(nodeLinks.size())) {
            continue;  // 방어적 가드 (불변식상 도달 불가)
        }
        for (const int32_t nb : nodeLinks[static_cast<size_t>(level)]) {
            if (visitedStamp_[static_cast<size_t>(nb)] == stamp_) {
                continue;
            }
            visitedStamp_[static_cast<size_t>(nb)] = stamp_;
            const float d = distTo(q, nb);
            if (static_cast<int32_t>(result.size()) < ef || d < result.top().dist) {
                candidates.push({d, nb});
                result.push({d, nb});
                if (static_cast<int32_t>(result.size()) > ef) {
                    result.pop();
                }
            }
        }
    }

    std::vector<Cand> out;
    out.reserve(result.size());
    while (!result.empty()) {
        out.push_back(result.top());
        result.pop();
    }
    std::reverse(out.begin(), out.end());  // dist 오름차순
    return out;
}

void HnswIndex::linkNeighbors(int32_t node, int32_t level, const std::vector<Cand>& cands) {
    const int32_t cap = (level == 0) ? maxM0_ : m_;
    const int32_t take = std::min(m_, static_cast<int32_t>(cands.size()));

    auto& mine = links_[static_cast<size_t>(node)][static_cast<size_t>(level)];
    mine.clear();
    mine.reserve(static_cast<size_t>(take));
    for (int32_t i = 0; i < take; ++i) {
        mine.push_back(cands[static_cast<size_t>(i)].node);
    }

    // 역방향 연결 + 용량 초과 시 가까운 cap 개만 유지
    for (int32_t i = 0; i < take; ++i) {
        const int32_t nb = cands[static_cast<size_t>(i)].node;
        auto& theirs = links_[static_cast<size_t>(nb)][static_cast<size_t>(level)];
        theirs.push_back(node);
        if (static_cast<int32_t>(theirs.size()) > cap) {
            const float* nbRow = row(nb);
            std::sort(theirs.begin(), theirs.end(), [&](int32_t a, int32_t b) {
                return distTo(nbRow, a) < distTo(nbRow, b);
            });
            theirs.resize(static_cast<size_t>(cap));
        }
    }
}

void HnswIndex::add(int32_t id, const float* vec) {
    const int32_t node = static_cast<int32_t>(ids_.size());
    ids_.push_back(id);
    const size_t offset = data_.size();
    data_.insert(data_.end(), vec, vec + dim_);
    l2NormalizeInPlace(data_.data() + offset, dim_);
    const float* q = row(node);

    const int32_t level = randomLevel();
    links_.emplace_back();
    links_.back().resize(static_cast<size_t>(level) + 1);

    if (entry_ < 0) {
        entry_ = node;
        maxLevel_ = level;
        return;
    }

    // 1) 최상층부터 목표 레벨+1 까지 greedy 하강
    int32_t cur = entry_;
    for (int32_t lvl = maxLevel_; lvl > level; --lvl) {
        cur = greedyClosest(q, cur, lvl);
    }
    // 2) 목표 레벨부터 0 까지: beam 탐색 → 이웃 연결
    for (int32_t lvl = std::min(level, maxLevel_); lvl >= 0; --lvl) {
        const std::vector<Cand> cands = searchLayer(q, cur, lvl, efConstruction_);
        if (!cands.empty()) {
            cur = cands.front().node;
        }
        linkNeighbors(node, lvl, cands);
    }
    // 3) 새 노드가 더 높은 레벨이면 entry 갱신
    if (level > maxLevel_) {
        maxLevel_ = level;
        entry_ = node;
    }
}

std::vector<SearchHit> HnswIndex::topK(const float* query, int32_t k) const {
    const int32_t n = static_cast<int32_t>(ids_.size());
    if (k <= 0 || n == 0) {
        return {};
    }

    std::vector<float> q(query, query + dim_);
    l2NormalizeInPlace(q.data(), dim_);

    int32_t cur = entry_;
    for (int32_t lvl = maxLevel_; lvl >= 1; --lvl) {
        cur = greedyClosest(q.data(), cur, lvl);
    }
    const int32_t ef = std::max(efSearch_, k);
    const std::vector<Cand> cands = searchLayer(q.data(), cur, 0, ef);

    const int32_t kk = std::min(k, static_cast<int32_t>(cands.size()));
    std::vector<SearchHit> hits;
    hits.reserve(static_cast<size_t>(kk));
    for (int32_t i = 0; i < kk; ++i) {
        const Cand& c = cands[static_cast<size_t>(i)];
        hits.push_back({ids_[static_cast<size_t>(c.node)], 1.0f - c.dist});
    }
    return hits;
}

void HnswIndex::clear() {
    ids_.clear();
    data_.clear();
    links_.clear();
    entry_ = -1;
    maxLevel_ = -1;
    visitedStamp_.clear();
    stamp_ = 0;
}

bool HnswIndex::writeBody(std::FILE* f) const {
    if (!(io::writePod(f, m_) && io::writePod(f, efConstruction_) && io::writePod(f, efSearch_) &&
          io::writePod(f, entry_) && io::writePod(f, maxLevel_))) {
        return false;
    }
    if (!io::writeBytes(f, ids_.data(), ids_.size() * sizeof(int32_t)) ||
        !io::writeBytes(f, data_.data(), data_.size() * sizeof(float))) {
        return false;
    }
    for (const auto& nodeLinks : links_) {
        const int32_t numLevels = static_cast<int32_t>(nodeLinks.size());
        if (!io::writePod(f, numLevels)) {
            return false;
        }
        for (const auto& neighbors : nodeLinks) {
            const int32_t n = static_cast<int32_t>(neighbors.size());
            if (!io::writePod(f, n) ||
                !io::writeBytes(f, neighbors.data(), neighbors.size() * sizeof(int32_t))) {
                return false;
            }
        }
    }
    return true;
}

bool HnswIndex::readBody(std::FILE* f, int64_t count) {
    if (count < 0 || dim_ <= 0) {
        return false;
    }
    const size_t n = static_cast<size_t>(count);
    if (n > kMaxLoadFloats / static_cast<size_t>(dim_)) {
        return false;
    }

    int32_t m = 0, efc = 0, efs = 0, entry = -1, maxLevel = -1;
    if (!(io::readPod(f, &m) && io::readPod(f, &efc) && io::readPod(f, &efs) &&
          io::readPod(f, &entry) && io::readPod(f, &maxLevel))) {
        return false;
    }
    if (m < 2 || efc < m || efs < 1) {
        return false;
    }
    if (count == 0) {
        if (entry != -1 || maxLevel != -1) {
            return false;
        }
    } else if (entry < 0 || entry >= count || maxLevel < 0 || maxLevel >= kMaxLevels) {
        return false;
    }

    m_ = m;
    maxM0_ = 2 * m;
    efConstruction_ = efc;
    efSearch_ = efs;
    levelMult_ = 1.0 / std::log(static_cast<double>(m_));
    entry_ = entry;
    maxLevel_ = maxLevel;

    ids_.resize(n);
    data_.resize(n * static_cast<size_t>(dim_));
    if (!io::readBytes(f, ids_.data(), ids_.size() * sizeof(int32_t)) ||
        !io::readBytes(f, data_.data(), data_.size() * sizeof(float))) {
        return false;
    }

    links_.assign(n, {});
    for (size_t i = 0; i < n; ++i) {
        int32_t numLevels = 0;
        if (!io::readPod(f, &numLevels) || numLevels < 1 || numLevels > kMaxLevels) {
            return false;
        }
        links_[i].resize(static_cast<size_t>(numLevels));
        for (int32_t lvl = 0; lvl < numLevels; ++lvl) {
            int32_t neighborCount = 0;
            if (!io::readPod(f, &neighborCount) || neighborCount < 0 || neighborCount > maxM0_) {
                return false;
            }
            auto& neighbors = links_[i][static_cast<size_t>(lvl)];
            neighbors.resize(static_cast<size_t>(neighborCount));
            if (!io::readBytes(f, neighbors.data(), neighbors.size() * sizeof(int32_t))) {
                return false;
            }
            for (const int32_t nb : neighbors) {
                if (nb < 0 || nb >= count) {
                    return false;
                }
            }
        }
    }
    if (count > 0 &&
        static_cast<size_t>(maxLevel_) >= links_[static_cast<size_t>(entry_)].size()) {
        return false;
    }

    visitedStamp_.clear();
    stamp_ = 0;
    return true;
}

}  // namespace rag
