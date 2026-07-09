# OnDevice RAG — 완전 오프라인 온디바이스 RAG 라이브러리

[![CI](https://github.com/honey0127/Lib/actions/workflows/ci.yml/badge.svg?branch=honey)](https://github.com/honey0127/Lib/actions/workflows/ci.yml)

Android 앱에 **네트워크 없이 동작하는 AI 문서 검색(RAG)** 을 넣는 오픈소스 Kotlin 라이브러리.
성능 핵심 경로(벡터 유사도 검색)는 C++17로 구현하고 JNI로 연결한다.

> 2026 오픈소스 개발자대회 자유과제 · **AI · 보안/안전 트랙** 출품작

## 왜 온디바이스인가

- **프라이버시**: 문서·질문·검색 기록이 전부 기기 안에서 처리된다. 서버 전송 없음.
- **오프라인**: 비행기 모드에서도 동작한다. 망분리·현장 업무 환경에 적합.
- **비용 0**: API 키도, 서버 비용도 없다. 라이브러리(AAR) 하나로 앱에 내장.

## 아키텍처

```
Kotlin API
└─ RagEngine                 문서 추가(addDocument) / 검색(search) 파사드
   ├─ TextChunker            문서 → 청크 분할 (문장 경계 우선 + 오버랩)
   ├─ Embedder               텍스트 → 벡터 인터페이스 (embed=문서, embedQuery=질의)
   │   ├─ HashingEmbedder    무모델 기본값 (문자 n-gram 해싱 — 데모/테스트용)
   │   └─ OnnxEmbedder       ONNX Runtime + multilingual-e5-small (시맨틱 검색)
   │       └─ UnigramTokenizer  SentencePiece Unigram 순수 Kotlin 구현 (Viterbi)
   └─ NativeVectorIndex      JNI 브리지 (FloatArray ↔ C++, librag-core.so)
       ├─ BruteForceIndex    정확 전수 비교 (수천 청크까지 기본값)
       └─ HnswIndex          HNSW 근사 최근접 이웃 (대규모 코퍼스, recall@10≈1.0)
└─ RagChat = RagEngine(검색) + LlmEngine(생성)
   └─ LlmEngine              llama.cpp(b9893, 정적 링크) + GGUF — 검색 근거로 한국어 답변 생성
```

## 사용 예

```kotlin
// 1) 무모델 기본값 (다운로드 0, 표면 유사도)
RagEngine().use { rag ->
    rag.addDocument("guide", "김치는 발효 음식이다. 배추와 고춧가루로 만든다.")
    rag.addDocument("space", "화성은 태양계의 네 번째 행성이다.")

    val results = rag.search("발효 음식이 뭐야?", topK = 3)
    results.forEach { println("${it.score} ${it.chunk.docId}: ${it.chunk.text}") }
}

// 2) 시맨틱 검색 (scripts/prepare_model.py 로 모델 준비 후)
val embedder = OnnxEmbedder.create("/path/model.onnx", "/path/vocab.tsv")
RagEngine(embedder, indexKind = IndexKind.HNSW).use { rag -> /* ... */ }

// 3) 스냅샷 — 재시작 시 재인덱싱 없이 즉시 복원
rag.save(File(dir, "snapshot"))                  // snapshot.idx + snapshot.meta
val restored = RagEngine.load(File(dir, "snapshot"))

// 4) 답변 생성 (RAG 완성체) — 검색 근거로 LLM 이 한국어 답변 (백그라운드 스레드에서)
val llm = LlmEngine.create("/path/llm.gguf")
val answer = RagChat(rag, llm).ask("발효 음식이 뭐야?") { piece -> print(piece); true }
println(answer.sources)                          // 근거 청크 (출처 표시)
```

### LLM 모델 준비 (Phase 3)

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
adb push models/qwen2.5-1.5b-instruct-q4_k_m.gguf /data/local/tmp/rag-models/llm.gguf
```

### 데모 앱 실행 (Phase 4)

```bash
./gradlew :app:installDebug   # 기기/에뮬레이터에 데모 앱 설치
```

모델 파일이 `/data/local/tmp/rag-models/` 에 있으면 자동으로 사용한다
(e5 ONNX → 시맨틱 검색, llm.gguf → 답변 생성). 없으면 무모델 폴백(해싱 임베더, 검색만)으로도 동작한다 —
"샘플 문서 4개 넣기 → 질문" 만으로 오프라인 검색 데모가 된다.

## 빌드 & 검증

사전 조건: Android SDK/NDK, JDK 21.

```bash
# ⓪ C++ 코어 로직만 빠르게 검증 (Android 불필요 — 아무 PC에서나)
g++ -std=c++17 -O2 rag-library/src/main/cpp/vector_index.cpp \
    rag-library/src/main/cpp/tests/vector_index_test.cpp -o /tmp/vt && /tmp/vt

# ① C++17 컴파일 + CMake 링킹 (librag-core.so)
./gradlew :rag-library:externalNativeBuildDebug

# ② 유닛테스트 + AAR 산출
./gradlew :rag-library:testDebugUnitTest :rag-library:assembleRelease

# ③ 실기기/에뮬레이터에서 JNI 왕복 + 검색 E2E 검증 (핵심 게이트)
./gradlew :rag-library:connectedDebugAndroidTest
```

기기가 없어도 GitHub Actions CI가 push마다 ⓪①②③(③은 에뮬레이터)을 자동 실행한다.

## 로드맵

| 단계 | 내용 | 상태 |
|---|---|---|
| Phase 0 | JNI 뼈대 + C++ 코사인 top-k + 청킹/검색 파이프라인 | ✅ |
| Phase 0.5 | ONNX Runtime + e5 임베딩 + 순수 Kotlin SentencePiece 토크나이저 | ✅ 코드 완료 (실기기 검증 대기) |
| Phase 1 | C++ HNSW 벡터 인덱스 (`IndexKind.HNSW`) | ✅ (CI 계측 통과, recall 1.0) |
| Phase 2 | 인덱스 영속화 — `RagEngine.save()/load()` 스냅샷 | ✅ 코드 완료 |
| Phase 3 | LLM 답변 생성 — llama.cpp(b9893) + Qwen GGUF, `RagChat` 스트리밍 | ✅ 코드 완료 (실기기 검증 대기) |
| Phase 4 | 데모 앱 UI (`:app`) — 문서 추가/검색/스트리밍 답변 | ✅ 코드 완료 |

## 성능

핵심 검색 루프는 SIMD(내적: arm64 NEON / x86_64 SSE2)로 벡터화되어 있다.
호스트 벤치마크(dim=384, 2만 청크, 무작위 데이터 = 최악 조건, `cpp/tests/bench.cpp`):

| 항목 | 스칼라 | SIMD 적용 후 |
|---|---|---|
| 브루트포스 검색 | 8.79 ms/질의 | **1.68 ms/질의** (5.2×) |
| HNSW 검색 | 1.49 ms/질의 | **0.48 ms/질의** |

HNSW 재현율은 동일 데이터·파라미터에서 레퍼런스 구현(hnswlib)과 ±0.02 이내로 일치
(ef 100~800 스윕) — 구현 정확성의 근거로 삼는다. 이웃 선택은 원논문 Algorithm 4
다양성 휴리스틱을 사용한다.

## 모델 가중치 정책

모델 가중치(`*.onnx`, `*.gguf` 등)는 재배포 라이선스와 용량 문제로 **저장소에 포함하지 않는다**.
앱 최초 실행 시 원 출처에서 다운로드하는 설계다. 출처·라이선스는 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 참고.

## 라이선스 및 대회 규정 준수

- 직접 작성한 전체 소스코드: [Apache License 2.0](LICENSE) (OSI 인증) + [NOTICE](NOTICE)
- 서드파티 라이브러리·모델 출처/라이선스: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- AI 모델 활용 명세(운영규정 제9조 대응 — 오픈웨이트 이상·온디바이스 구동·API 의존 없음):
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 하단 표 참고
- 모델 가중치는 저장소에 포함하지 않고 원 출처(Hugging Face)에서 내려받는다
