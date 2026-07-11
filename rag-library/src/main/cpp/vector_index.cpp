#include "vector_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

#include "dot_product.h"
#include "index_io.h"

namespace rag {

namespace {
constexpr double kNormEps = 1e-12;
// 파일 로드 시 허용하는 최대 원소 수 (손상 파일의 과대 할당 방지, ~1GB)
constexpr size_t kMaxLoadFloats = size_t(1) << 28;
}

void l2NormalizeInPlace(float* v, int32_t dim) {
    double sum = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    const double norm = std::sqrt(sum);
    if (norm < kNormEps) {
        std::memset(v, 0, static_cast<size_t>(dim) * sizeof(float));
        return;
    }
    const float inv = static_cast<float>(1.0 / norm);
    for (int32_t i = 0; i < dim; ++i) {
        v[i] *= inv;
    }
}

BruteForceIndex::BruteForceIndex(int32_t dim) : dim_(dim) {}

void BruteForceIndex::add(int32_t id, const float* vec) {
    ids_.push_back(id);
    const size_t offset = data_.size();
    data_.insert(data_.end(), vec, vec + dim_);
    l2NormalizeInPlace(data_.data() + offset, dim_);
}

std::vector<SearchHit> BruteForceIndex::topK(const float* query, int32_t k,
                                             const uint8_t* allowMask, int32_t maskLen) const {
    const int32_t n = static_cast<int32_t>(ids_.size());
    const int32_t kk = std::min(k, n);
    if (kk <= 0) {
        return {};
    }

    std::vector<float> q(query, query + dim_);
    l2NormalizeInPlace(q.data(), dim_);

    // 크기 kk 의 min-heap(최저 점수가 top) 유지 — 전체 n 개 후보 배열을 만들지 않는다.
    const auto worstOnTop = [](const SearchHit& a, const SearchHit& b) {
        return a.score > b.score;
    };
    std::vector<SearchHit> heap;
    heap.reserve(static_cast<size_t>(kk));
    for (int32_t i = 0; i < n; ++i) {
        const int32_t id = ids_[static_cast<size_t>(i)];
        if (allowMask != nullptr && (id < 0 || id >= maskLen || allowMask[id] == 0)) {
            continue;  // 스캔 루프 내 필터 — 정확도 손실 0
        }
        const float* row = data_.data() + static_cast<size_t>(i) * dim_;
        const float dot = dotProduct(row, q.data(), dim_);
        if (static_cast<int32_t>(heap.size()) < kk) {
            heap.push_back({id, dot});
            std::push_heap(heap.begin(), heap.end(), worstOnTop);
        } else if (dot > heap.front().score) {
            std::pop_heap(heap.begin(), heap.end(), worstOnTop);
            heap.back() = {id, dot};
            std::push_heap(heap.begin(), heap.end(), worstOnTop);
        }
    }
    std::sort_heap(heap.begin(), heap.end(), worstOnTop);  // 점수 내림차순 정렬로 마무리
    return heap;
}

size_t BruteForceIndex::removeByIds(const int32_t* removeIds, size_t count) {
    if (count == 0 || ids_.empty()) {
        return 0;
    }
    std::unordered_set<int32_t> target(removeIds, removeIds + count);
    size_t removed = 0;
    size_t i = 0;
    while (i < ids_.size()) {
        if (target.count(ids_[i]) == 0) {
            ++i;
            continue;
        }
        // swap-remove: 마지막 행을 i 로 옮기고 꼬리를 잘라낸다 (연속 메모리풀 유지)
        const size_t last = ids_.size() - 1;
        if (i != last) {
            ids_[i] = ids_[last];
            std::memcpy(data_.data() + i * static_cast<size_t>(dim_),
                        data_.data() + last * static_cast<size_t>(dim_),
                        static_cast<size_t>(dim_) * sizeof(float));
        }
        ids_.pop_back();
        data_.resize(data_.size() - static_cast<size_t>(dim_));
        ++removed;
        // i 자리에 새 행이 왔으므로 i 는 전진하지 않는다
    }
    return removed;
}

void BruteForceIndex::clear() {
    ids_.clear();
    data_.clear();
}

bool BruteForceIndex::writeBody(std::FILE* f) const {
    return io::writeBytes(f, ids_.data(), ids_.size() * sizeof(int32_t)) &&
           io::writeBytes(f, data_.data(), data_.size() * sizeof(float));
}

bool BruteForceIndex::readBody(std::FILE* f, int64_t count) {
    if (count < 0 || dim_ <= 0) {
        return false;
    }
    const size_t n = static_cast<size_t>(count);
    if (n > kMaxLoadFloats / static_cast<size_t>(dim_)) {
        return false;
    }
    ids_.resize(n);
    data_.resize(n * static_cast<size_t>(dim_));
    return io::readBytes(f, ids_.data(), ids_.size() * sizeof(int32_t)) &&
           io::readBytes(f, data_.data(), data_.size() * sizeof(float));
}

}  // namespace rag
