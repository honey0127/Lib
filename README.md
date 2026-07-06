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
```

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
| Phase 3 | LLM(GGUF, llama.cpp) 통합 — 검색 결과 기반 답변 생성 | ⏳ |
| Phase 4 | 데모 앱 UI (`:app`) | ⏳ |

## 모델 가중치 정책

모델 가중치(`*.onnx`, `*.gguf` 등)는 재배포 라이선스와 용량 문제로 **저장소에 포함하지 않는다**.
앱 최초 실행 시 원 출처에서 다운로드하는 설계다. 출처·라이선스는 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) 참고.

## 라이선스

[Apache License 2.0](LICENSE) · 서드파티: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
