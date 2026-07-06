#include "index_io.h"

#include <cstdint>
#include <cstdio>
#include <memory>

#include "hnsw_index.h"
#include "vector_index.h"

namespace rag {

namespace {

constexpr uint32_t kMagic = 0x31474152u;  // little-endian 파일에서 "RAG1"

struct FileCloser {
    void operator()(std::FILE* f) const {
        if (f != nullptr) {
            std::fclose(f);
        }
    }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

}  // namespace

bool saveIndex(const VectorIndex& idx, const char* path) {
    FilePtr f(std::fopen(path, "wb"));
    if (!f) {
        return false;
    }
    const int64_t count = static_cast<int64_t>(idx.size());
    bool ok = io::writePod(f.get(), kMagic) && io::writePod(f.get(), idx.formatKind()) &&
              io::writePod(f.get(), idx.dim()) && io::writePod(f.get(), count) &&
              idx.writeBody(f.get());
    if (ok) {
        ok = (std::fflush(f.get()) == 0);
    }
    f.reset();
    if (!ok) {
        std::remove(path);  // 불완전 파일 잔류 방지
    }
    return ok;
}

std::unique_ptr<VectorIndex> loadIndex(const char* path) {
    FilePtr f(std::fopen(path, "rb"));
    if (!f) {
        return nullptr;
    }
    uint32_t magic = 0;
    uint32_t kind = 0;
    int32_t dim = 0;
    int64_t count = 0;
    if (!(io::readPod(f.get(), &magic) && io::readPod(f.get(), &kind) &&
          io::readPod(f.get(), &dim) && io::readPod(f.get(), &count))) {
        return nullptr;
    }
    if (magic != kMagic || dim <= 0 || dim > (1 << 16)) {
        return nullptr;
    }

    std::unique_ptr<VectorIndex> idx;
    if (kind == BruteForceIndex::kFormatKind) {
        idx = std::make_unique<BruteForceIndex>(dim);
    } else if (kind == HnswIndex::kFormatKind) {
        idx = std::make_unique<HnswIndex>(dim);
    } else {
        return nullptr;
    }
    if (!idx->readBody(f.get(), count)) {
        return nullptr;
    }
    return idx;
}

}  // namespace rag
