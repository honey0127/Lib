// LlmEngine ↔ llama.cpp(b9893) 통합 스모크 테스트 (호스트 전용, CI 미포함 — llama.cpp 빌드 필요).
//
// llama.cpp 저장소의 vocab 전용 GGUF(가중치 없음, 수백 KB)로 로드/토크나이즈 경로와
// UTF-8 경계 헬퍼를 검증한다. 실제 생성(generate)은 실기기 + 실제 모델에서 확인.
//
// 실행 방법 (개발 PC — 각 명령은 한 줄로 이어서 실행):
//   git clone --depth 1 --branch b9893 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
//   cmake -S /tmp/llama.cpp -B /tmp/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release
//     -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
//     -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF
//     -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
//   ninja -C /tmp/llama-build llama
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror
//     -I/tmp/llama.cpp/include -I/tmp/llama.cpp/ggml/include
//     rag-library/src/main/cpp/llm_engine.cpp
//     rag-library/src/main/cpp/tests/llm_engine_smoke.cpp
//     /tmp/llama-build/src/libllama.a /tmp/llama-build/ggml/src/libggml.a
//     /tmp/llama-build/ggml/src/libggml-cpu.a /tmp/llama-build/ggml/src/libggml-base.a
//     -lpthread -ldl -o /tmp/llm_smoke
//   /tmp/llm_smoke /tmp/llama.cpp/models/ggml-vocab-qwen2.gguf

#include "../llm_engine.h"

#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // 1) UTF-8 경계 헬퍼 — 순수 함수라 모델 없이 검증
    {
        using rag::completeUtf8Prefix;
        expect(completeUtf8Prefix("") == 0, "empty -> 0");
        expect(completeUtf8Prefix("abc") == 3, "ascii complete");
        const std::string kim = "\xEA\xB9\x80";  // '김' (3바이트)
        expect(completeUtf8Prefix(kim) == 3, "complete hangul");
        expect(completeUtf8Prefix(kim.substr(0, 2)) == 0, "truncated hangul -> 0");
        expect(completeUtf8Prefix("ab" + kim.substr(0, 1)) == 2, "ascii + lead byte -> 2");
        const std::string emoji = "\xF0\x9F\x99\x82";  // 🙂 (4바이트)
        expect(completeUtf8Prefix(emoji) == 4, "complete emoji");
        expect(completeUtf8Prefix(emoji.substr(0, 3)) == 0, "truncated emoji -> 0");
        expect(completeUtf8Prefix(kim + emoji.substr(0, 2)) == 3, "hangul + partial emoji -> 3");
    }

    // 2) vocab 전용 GGUF 로드 + 토크나이즈 (인자로 경로를 받았을 때만)
    if (argc > 1) {
        rag::LlmEngine::Params params;
        rag::LlmEngine* engine = rag::LlmEngine::create(argv[1], params, /*vocabOnly=*/true);
        expect(engine != nullptr, "vocab-only model loads");
        if (engine != nullptr) {
            expect(!engine->canGenerate(), "vocab-only cannot generate");
            const int32_t nKo = engine->countTokens("김치는 발효 음식이다.");
            const int32_t nEn = engine->countTokens("Hello world");
            std::printf("countTokens: ko=%d en=%d\n", nKo, nEn);
            expect(nKo > 0 && nEn > 0, "tokenize returns positive counts");

            // ChatML 특수 토큰이 통짜로 파싱되는지 (qwen2 vocab 에 존재)
            const int32_t nPlain = engine->countTokens("질문");
            const int32_t nChat = engine->countTokens("<|im_start|>user\n질문<|im_end|>");
            std::printf("countTokens: plain=%d chatml=%d\n", nPlain, nChat);
            expect(nChat > nPlain && nChat <= nPlain + 6, "ChatML specials parse as few tokens");

            expect(engine->generate("x", 8, nullptr).empty(), "generate on vocab-only -> empty");
            delete engine;
        }
    } else {
        std::printf("(모델 경로 미지정 — UTF-8 헬퍼만 검증)\n");
    }

    if (g_failures == 0) {
        std::printf("llm_engine_smoke: ALL OK\n");
        return 0;
    }
    std::printf("llm_engine_smoke: %d FAILURE(S)\n", g_failures);
    return 1;
}
