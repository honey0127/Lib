#include "llm_engine.h"

#include <algorithm>
#include <vector>

#include "llama.h"

namespace rag {

namespace {

// llama 백엔드는 프로세스당 1회만 초기화
void ensureBackendInit() {
    static const bool once = [] {
        llama_backend_init();
        return true;
    }();
    (void)once;
}

}  // namespace

size_t completeUtf8Prefix(const std::string& buf) {
    const size_t n = buf.size();
    size_t i = n;
    size_t back = 0;
    while (i > 0 && back < 4) {
        --i;
        ++back;
        const unsigned char c = static_cast<unsigned char>(buf[i]);
        if ((c & 0x80u) == 0) {
            return n;  // 마지막으로 훑은 바이트가 ASCII → 전부 완전
        }
        if ((c & 0xC0u) == 0xC0u) {  // 리드 바이트 발견
            size_t need = 0;
            if ((c & 0xE0u) == 0xC0u) {
                need = 2;
            } else if ((c & 0xF0u) == 0xE0u) {
                need = 3;
            } else if ((c & 0xF8u) == 0xF0u) {
                need = 4;
            } else {
                return n;  // 비정상 바이트 — 그대로 내보냄
            }
            return (n - i >= need) ? n : i;  // 시퀀스가 완성됐으면 전부, 아니면 리드 앞까지
        }
        // continuation(10xxxxxx) → 계속 뒤로
    }
    return n;  // 리드 바이트를 못 찾음(비정상) — 그대로 내보냄
}

LlmEngine* LlmEngine::create(const char* modelPath, const Params& params, bool vocabOnly) {
    ensureBackendInit();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = vocabOnly;
    llama_model* model = llama_model_load_from_file(modelPath, mp);
    if (model == nullptr) {
        return nullptr;
    }

    auto* engine = new LlmEngine();
    engine->model_ = model;
    engine->vocab_ = llama_model_get_vocab(model);

    if (!vocabOnly) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(params.nCtx);
        cp.n_batch = static_cast<uint32_t>(params.nCtx);  // 프롬프트를 한 번에 평가
        cp.n_threads = params.nThreads;
        cp.n_threads_batch = params.nThreads;
        engine->ctx_ = llama_init_from_model(model, cp);
        if (engine->ctx_ == nullptr) {
            delete engine;
            return nullptr;
        }
        engine->nCtx_ = static_cast<int32_t>(llama_n_ctx(engine->ctx_));
        engine->nBatch_ = static_cast<int32_t>(llama_n_batch(engine->ctx_));

        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.topK));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.topP, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));
        engine->sampler_ = smpl;
    }
    return engine;
}

LlmEngine::~LlmEngine() {
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
    }
}

int32_t LlmEngine::countTokens(const std::string& textUtf8) const {
    // n_tokens_max=0 → 필요한 토큰 수의 음수를 반환하는 규약
    const int32_t neg = llama_tokenize(vocab_, textUtf8.c_str(),
                                       static_cast<int32_t>(textUtf8.size()), nullptr, 0,
                                       /*add_special=*/true, /*parse_special=*/true);
    return neg < 0 ? -neg : neg;
}

std::string LlmEngine::generate(const std::string& promptUtf8, int32_t maxTokens,
                                const std::function<bool(const std::string&)>& onPiece) {
    if (ctx_ == nullptr || maxTokens <= 0) {
        return {};
    }

    // 1) 토크나이즈 (parse_special=true — ChatML <|im_start|> 등 특수 토큰 인식)
    std::vector<llama_token> tokens(promptUtf8.size() + 8);
    int32_t n = llama_tokenize(vocab_, promptUtf8.c_str(),
                               static_cast<int32_t>(promptUtf8.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), true, true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab_, promptUtf8.c_str(),
                           static_cast<int32_t>(promptUtf8.size()), tokens.data(),
                           static_cast<int32_t>(tokens.size()), true, true);
    }
    if (n <= 0 || n >= nCtx_) {
        return {};  // 빈 프롬프트 또는 컨텍스트 초과(호출측에서 청크 수 조절)
    }
    tokens.resize(static_cast<size_t>(n));

    // 2) 새 대화 — KV 캐시 비우고 프롬프트 평가 (n_batch 단위)
    llama_memory_clear(llama_get_memory(ctx_), true);
    for (int32_t off = 0; off < n; off += nBatch_) {
        const int32_t chunk = std::min(nBatch_, n - off);
        llama_batch batch = llama_batch_get_one(tokens.data() + off, chunk);
        if (llama_decode(ctx_, batch) != 0) {
            return {};
        }
    }

    // 3) 생성 루프 — EOG/예산/디코드 실패/콜백 중단에서 종료
    std::string out;
    std::string pending;  // UTF-8 경계 미완성 바이트
    char piece[256];
    const int32_t budget = std::min(maxTokens, nCtx_ - n);
    for (int32_t i = 0; i < budget; ++i) {
        llama_token tok = llama_sampler_sample(sampler_, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, tok)) {
            break;
        }

        const int32_t len = llama_token_to_piece(vocab_, tok, piece,
                                                 static_cast<int32_t>(sizeof(piece)), 0, false);
        if (len > 0) {
            pending.append(piece, static_cast<size_t>(len));
            const size_t complete = completeUtf8Prefix(pending);
            if (complete > 0) {
                const std::string ready = pending.substr(0, complete);
                pending.erase(0, complete);
                out += ready;
                if (onPiece && !onPiece(ready)) {
                    break;
                }
            }
        }

        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx_, batch) != 0) {
            break;
        }
    }
    // 남은 미완성 멀티바이트 조각은 버린다 (깨진 문자 방지)
    return out;
}

}  // namespace rag
