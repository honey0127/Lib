#pragma once

#include <cstdint>
#include <functional>
#include <string>

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;

namespace rag {

// UTF-8 버퍼에서 "완전한 문자"까지의 바이트 길이를 돌려준다.
// (토큰 조각이 멀티바이트 문자 중간에서 잘릴 수 있어 스트리밍 콜백 전에 경계를 맞춘다)
size_t completeUtf8Prefix(const std::string& buf);

// llama.cpp(GGUF) 온디바이스 텍스트 생성 엔진 (Phase 3).
// - llama.cpp 은 CMake FetchContent 로 태그 b9893 에 고정되어 정적 링크된다.
// - 스레드 안전하지 않다. 동기화는 호출측(Kotlin LlmEngine)이 담당.
// - generate() 는 호출마다 KV 캐시를 비우고 프롬프트를 새로 평가한다(단발 Q&A 용).
class LlmEngine {
public:
    struct Params {
        int32_t nCtx = 2048;       // 컨텍스트 토큰 수
        int32_t nThreads = 4;      // 생성 스레드 수
        float temperature = 0.7f;
        int32_t topK = 40;
        float topP = 0.9f;
        uint32_t seed = 42;
    };

    // 실패 시 nullptr.
    // vocabOnly=true 는 가중치 없이 토크나이저만 로드(호스트 스모크 테스트용, generate 불가).
    static LlmEngine* create(const char* modelPath, const Params& params, bool vocabOnly = false);
    ~LlmEngine();

    LlmEngine(const LlmEngine&) = delete;
    LlmEngine& operator=(const LlmEngine&) = delete;

    // 프롬프트(UTF-8, ChatML 특수 토큰 파싱됨) → 생성 텍스트(UTF-8).
    // onPiece 가 완전한 UTF-8 조각을 받는다. false 반환 시 생성 중단.
    // EOG 토큰·maxTokens·컨텍스트 한계에서 중단. 실패/vocabOnly 면 빈 문자열.
    std::string generate(const std::string& promptUtf8, int32_t maxTokens,
                         const std::function<bool(const std::string&)>& onPiece);

    // add_special/parse_special 포함 토큰 수 (프롬프트 예산 계산·스모크 테스트용)
    int32_t countTokens(const std::string& textUtf8) const;

    bool canGenerate() const { return ctx_ != nullptr; }

private:
    LlmEngine() = default;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    int32_t nCtx_ = 0;
    int32_t nBatch_ = 0;
};

}  // namespace rag
