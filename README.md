# OnDevice RAG — 완전 오프라인 온디바이스 RAG 라이브러리

[![CI](https://github.com/honey0127/Lib/actions/workflows/ci.yml/badge.svg?branch=honey)](https://github.com/honey0127/Lib/actions/workflows/ci.yml)

Android 앱에 **네트워크 없이 동작하는 AI 문서 검색(RAG)** 을 넣는 오픈소스 Kotlin 라이브러리.
성능 핵심 경로(벡터 유사도 검색)는 C++17로 구현하고 JNI로 연결한다.

> 2026 오픈소스 개발자대회 자유과제 · **AI · 보안/안전 트랙** 출품작

## 설치와 10줄 통합

```kotlin
// settings.gradle.kts
dependencyResolutionManagement { repositories { maven("https://jitpack.io") } }
// 모듈 build.gradle.kts  (태그 릴리스 기준 — Releases 탭 참고)
dependencies { implementation("com.github.honey0127.Lib:rag-library:<tag>") }
```

```kotlin
RagEngine().use { rag ->                                   // 1) 모델 없이 즉시 동작
    rag.addDocument("guide", "김치는 발효 음식이다. 배추로 만든다.")
    val hits = rag.search("발효 음식", topK = 3)            // 2) 완전 오프라인 검색
    rag.search("발효", topK = 3, allowDocIds = setOf("guide")) // 문서 필터 (정확도 손실 0)
    rag.removeDocument("guide")                             // 문서 삭제 (swap-remove)
    hits.forEach { println("${it.score}  ${it.chunk.text}") }
}
// 3) 시맨틱 검색 + 스트리밍 답변 (모델 파일 배치 후 — 아래 '모델 준비')
val rag = RagEngine(OnnxEmbedder.create(onnxPath, vocabPath), indexKind = IndexKind.HNSW)
val chat = RagChat(rag, LlmEngine.create(ggufPath))
val answer = chat.ask("김치는 어떤 음식이야?") { piece -> print(piece); true }
println(answer.sources)                                    // 근거 청크(환각 방지 출처)
```

데모 앱(`:app`)이 그대로 "라이브러리 소비자 예제"다 — 위 API 만으로 화면 전체가 구성된다.

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

## 성능 — 측정으로 증명한다

핵심 검색 루프는 직접 작성한 SIMD 내적(arm64 NEON / x86_64 SSE2, `cpp/dot_product.h`)을 쓴다.
모든 수치는 저장소의 벤치 하네스로 재현 가능하다 (호스트 x86_64 · dim=384 · 무작위 데이터 = 최악 조건):

```bash
g++ -std=c++17 -O2 rag-library/src/main/cpp/vector_index.cpp \
    rag-library/src/main/cpp/hnsw_index.cpp \
    rag-library/src/main/cpp/tests/bench.cpp -o /tmp/rag_bench && /tmp/rag_bench
```

**① SIMD 내적 — 동일 바이너리 내 직접 비교 (2천만 회 호출)**

| dot(384) | 스칼라 | SIMD | 배율 |
|---|---|---|---|
| 총 시간 | 43.2 ms | **10.6 ms** | **4.1×** |

**② 검색 지연 — 코퍼스 규모 스윕 (ms/질의, top-10)**

| 코퍼스 크기 | 브루트포스(정확) | HNSW(근사, ef=100) |
|---|---|---|
| 1,000 | 0.06 | — |
| 10,000 | 0.62 | — |
| 20,000 | 1.67 | **0.43** |
| 50,000 | 7.06 | (규모↑일수록 격차 확대) |

**③ HNSW 정확성 — 레퍼런스(hnswlib)와 동일 데이터·파라미터 교차 검증**

ef 100/200/400/800 스윕에서 재현율이 hnswlib 와 전 구간 ±0.02 이내 동률
(0.352/0.565/0.779/0.923 vs 0.365/0.582/0.793/0.935). 무작위 고차원은 거리 집중으로
재현율이 낮게 나오는 최악 조건이며, 실제 임베딩(구조 있음)에선 같은 ef 에서 훨씬 높다.
이웃 선택은 원논문(Malkov) Algorithm 4 다양성 휴리스틱 자체 구현.

**③.5 인덱스 선택은 규모의 함수다 — 기본값이 브루트포스인 이유**

모바일 개인 문서 규모(수천~수만 청크)에선 SIMD 브루트포스가 1.7ms/질의(2만 청크)로 충분히
빠르면서, **docId 필터링이 스캔 루프 안에서 정확도 손실 0**으로 동작하고 **삭제가 swap-remove
로 무손실 공짜**다. 필터드 서치와 삭제가 알려진 난제인 ANN(HNSW) 계열과의 결정적 차이 —
단순해서가 아니라 이 규모에서 옳아서 기본값이다. HNSW(`IndexKind.HNSW`)는 수만 청크
이상에서 켜는 확장 옵션으로 제공한다(필터는 포스트 필터, 삭제는 재구축).

**④ JNI 경계 비용 — 어디를 최적화해야 하는지도 측정했다**

- 질의 경로: 384 float × 4B = **1.5KB 복사 → 검색 자체(0.4~7ms) 대비 무시 가능**.
  `GetPrimitiveArrayCritical`+즉시 해제 패턴으로 충분하다.
- 인덱싱 경로: C++ add 는 벡터당 2~16µs. 병목은 임베딩 추론(청크당 수 ms) 쪽이므로
  JNI 배치 API 는 측정상 불필요해 추가하지 않았다 — 최적화는 수치가 요구하는 곳에만.

## 모델 가중치 정책

모델 가중치(`*.onnx`, `*.gguf` 등)는 재배포 라이선스와 용량 문제로 **저장소에 포함하지 않는다**.
앱 최초 실행 시 원 출처에서 다운로드하는 설계다. 출처·라이선스는 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 참고.

## 라이선스 및 대회 규정 준수

직접 작성한 전체 소스코드는 [Apache License 2.0](LICENSE)(OSI 인증) + [NOTICE](NOTICE).
모델 가중치는 저장소에 포함하지 않고 원 출처(Hugging Face)에서 내려받는다.

| 구성요소 | 라이선스 | 사용 형태 |
|---|---|---|
| ONNX Runtime | MIT | 임베딩 추론 (Maven 의존성) |
| llama.cpp (b9893) | MIT | LLM 추론 (정적 링크) |
| multilingual-e5-small | MIT | 오픈웨이트 · 그대로 활용 · 온디바이스 |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | 오픈웨이트 · 그대로 활용 · 온디바이스 |
| AndroidX / Material | Apache-2.0 | 라이브러리/데모 |
| JUnit 4 | EPL-1.0 | 테스트 전용 |

세부 출처와 **AI 모델 활용 명세(운영규정 제9조 대응)**: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
— 외부 API 의존 0회(완전 온디바이스), 추론 코드 전체 자체 작성(Apache-2.0).
