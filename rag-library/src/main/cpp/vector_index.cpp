#include "vector_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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

std::vector<SearchHit> BruteForceIndex::topK(const float* query, int32_t k) const {
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
        const float* row = data_.data() + static_cast<size_t>(i) * dim_;
        const float dot = dotProduct(row, q.data(), dim_);
        if (static_cast<int32_t>(heap.size()) < kk) {
            heap.push_back({ids_[i], dot});
            std::push_heap(heap.begin(), heap.end(), worstOnTop);
        } else if (dot > heap.front().score) {
            std::pop_heap(heap.begin(), heap.end(), worstOnTop);
            heap.back() = {ids_[i], dot};
            std::push_heap(heap.begin(), heap.end(), worstOnTop);
        }
    }
    std::sort_heap(heap.begin(), heap.end(), worstOnTop);  // 점수 내림차순 정렬로 마무리
    return heap;
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
