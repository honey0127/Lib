#include "vector_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rag {

namespace {

constexpr double kNormEps = 1e-12;

// v 를 제자리 L2 정규화. 노름이 0에 가까우면 0 벡터로 만든다(NaN 방지).
void normalizeInPlace(float* v, int32_t dim) {
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

}  // namespace

VectorIndex::VectorIndex(int32_t dim) : dim_(dim) {}

void VectorIndex::add(int32_t id, const float* vec) {
    ids_.push_back(id);
    const size_t offset = data_.size();
    data_.insert(data_.end(), vec, vec + dim_);
    normalizeInPlace(data_.data() + offset, dim_);
}

std::vector<SearchHit> VectorIndex::topK(const float* query, int32_t k) const {
    const int32_t n = static_cast<int32_t>(ids_.size());
    const int32_t kk = std::min(k, n);
    if (kk <= 0) {
        return {};
    }

    std::vector<float> q(query, query + dim_);
    normalizeInPlace(q.data(), dim_);

    std::vector<SearchHit> hits;
    hits.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        const float* row = data_.data() + static_cast<size_t>(i) * dim_;
        float dot = 0.0f;
        for (int32_t j = 0; j < dim_; ++j) {
            dot += row[j] * q[j];
        }
        hits.push_back({ids_[i], dot});
    }

    std::partial_sort(hits.begin(), hits.begin() + kk, hits.end(),
                      [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    hits.resize(static_cast<size_t>(kk));
    return hits;
}

void VectorIndex::clear() {
    ids_.clear();
    data_.clear();
}

}  // namespace rag
