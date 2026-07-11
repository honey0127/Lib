# CLAUDE.md — 온디바이스 RAG 라이브러리

Android 앱에 **완전 오프라인** AI 문서 검색을 넣는 오픈소스 Kotlin 라이브러리. 성능 핵심 경로는 C++/JNI.
(2026 오픈소스 개발자대회 자유과제 · AI·보안/안전 트랙 출품작)

이 문서는 Claude Code / 개발자를 위한 **빌드·검증 명령 및 JNI 규칙의 단일 진실 소스**다.
반복 빌드-수정 루프에서 먼저 참조한다.

**작업 브랜치: `honey`** (사용자 지시 — 직접 이 브랜치에 커밋·푸시한다.)

## 모듈 구조
- `:rag-library` — 라이브러리 본체(AAR). Kotlin API + `src/main/cpp` 의 C++/JNI 코어(`librag-core.so`). **개발의 중심.**
- `:app` — 데모 앱(namespace `com.example.lib`). `MainActivity` 단일 화면: 문서 추가/샘플 주입(1회)
  → 질문 → 근거(유사도) + LLM 스트리밍 답변 + 처리시간(ms) 표시. 초기화/저장/불러오기 버튼으로
  RagEngine.clear/save/load 시연. 모델 파일은 `/data/local/tmp/rag-models/` 에서 자동 감지,
  없으면 무모델 폴백(해싱 임베더·검색만 — 같은 언어 표면 매칭임을 상태줄에 안내).
  엔진 접근은 단일 스레드 executor 로 직렬화.

```
rag-library/src/main/
├── AndroidManifest.xml
├── cpp/
│   ├── CMakeLists.txt         # add_library(rag-core SHARED native-lib rag_jni vector_index hnsw_index index_io)
│   ├── native-lib.cpp         # Phase 0 JNI 왕복 스모크(stringFromJNI)
│   ├── dot_product.h          # SIMD 내적 (NEON/SSE2/스칼라) — 검색 핫루프
│   ├── vector_index.{h,cpp}   # VectorIndex 추상 인터페이스 + BruteForceIndex + l2NormalizeInPlace
│   ├── hnsw_index.{h,cpp}     # HnswIndex — HNSW + 다양성 휴리스틱(Algorithm 4) (Phase 1)
│   ├── index_io.{h,cpp}       # saveIndex/loadIndex — 'RAG1' 단일 바이너리 직렬화 (Phase 2)
│   ├── llm_engine.{h,cpp}     # llama.cpp 래퍼 — 로드/토크나이즈/생성 + UTF-8 경계 (Phase 3)
│   ├── rag_jni.cpp            # NativeVectorIndex JNI 브리지 (핸들 기반 10개 함수)
│   ├── llm_jni.cpp            # NativeLlm JNI 브리지 (4개 함수, ByteArray UTF-8 왕복)
│   └── tests/                 # 호스트(g++) 단독 실행 테스트 — CMake 타깃에 미포함
│       ├── vector_index_test.cpp # + SIMD dotProduct 스칼라 대비 검증
│       ├── hnsw_index_test.cpp   # 소규모 정확성 + 브루트포스 대비 recall@10
│       ├── index_io_test.cpp     # 저장/복원 왕복 + 손상 파일 거부
│       ├── bench.cpp             # 성능 벤치마크 (수동 — 아래 '성능' 참고)
│       └── llm_engine_smoke.cpp  # llama.cpp 링크 + vocab GGUF 스모크 (수동, 파일 상단 명령 참고)
└── java/com/example/rag_library/
    ├── OnDeviceRAGCore.kt     # loadLibrary("rag-core") + stringFromJNI() (스모크)
    ├── NativeVectorIndex.kt   # C++ 인덱스 래퍼 — bruteForce()/hnsw()/load() + save()
    ├── ChunkStore.kt          # RagEngine 스냅샷 메타(청크) 직렬화 (순수 JVM)
    ├── Models.kt              # Chunk / SearchResult
    ├── TextChunker.kt         # 문장 경계 + overlap 청킹
    ├── Embedder.kt            # 임베딩 인터페이스 (embed=passage, embedQuery=질의)
    ├── HashingEmbedder.kt     # 문자 n-gram 해싱 임베더 (무모델 기본값)
    ├── Tokenizer.kt           # 토크나이저 인터페이스
    ├── UnigramTokenizer.kt    # SentencePiece Unigram 순수 Kotlin (Viterbi, vocab TSV)
    ├── Pooling.kt             # mean pooling + L2 정규화
    ├── OnnxEmbedder.kt        # ONNX Runtime + e5 (모델 파일은 scripts/prepare_model.py)
    ├── RagEngine.kt           # 공개 파사드: addDocument / search / IndexKind / save / load
    ├── NativeLlm.kt           # LLM JNI 브리지 (internal object, ByteArray UTF-8)
    ├── LlmEngine.kt           # 공개 LLM 래퍼 — create/generate(스트리밍)/countTokens
    ├── QwenPromptBuilder.kt   # ChatML RAG 프롬프트 (문서 근거 고정 시스템 프롬프트)
    └── RagChat.kt             # 공개 완성체: 검색 + 답변 생성 (ask + sources)
rag-library/src/test/…         # 호스트 유닛테스트 (Chunker/Hashing/Tokenizer/Pooling/ChunkStore/Prompt)
rag-library/src/androidTest/…  # 계측: JNI 왕복, 인덱스(브루트/HNSW), RagEngine E2E, 영속화,
                               #      OnnxEmbedderTest·LlmEngineTest(모델 파일 있을 때만, 없으면 skip)
scripts/prepare_model.py       # e5 ONNX 내보내기 + tokenizer.json→vocab.tsv (개발 PC용)
```

## 툴체인 (repo 실제 고정값)
| 항목 | 값 | 위치 |
|---|---|---|
| AGP | 9.2.1 | `gradle/libs.versions.toml` |
| Gradle | 9.4.1 | `gradle/wrapper/gradle-wrapper.properties` |
| Kotlin | AGP 9 built-in Kotlin (별도 플러그인 미적용, KGP 2.2.10+ 자동) | — |
| compileSdk | 36 (minorApiLevel 1) | 각 모듈 `build.gradle.kts` |
| minSdk | 26 | |
| Java | 11 (`kotlin.compilerOptions.jvmTarget`) | 각 모듈 |
| Gradle 데몬 JVM | 21 | `gradle/gradle-daemon-jvm.properties` |
| CMake | 3.22.1 | |
| NDK | SDK 기본값(미고정) | 재현성 위해 `ndkVersion` 고정 권장 |
| C++ | 17 | `cppFlags` / `CMakeLists.txt` |
| ABI | arm64-v8a, x86_64 | `rag-library` `defaultConfig.ndk.abiFilters` |
| llama.cpp | **태그 b9893 고정** (CMake FetchContent, 정적 링크) | `cpp/CMakeLists.txt` |

> llama.cpp 업그레이드 시: CMakeLists 의 태그만 바꾸지 말고, tests/llm_engine_smoke.cpp 를
> 새 태그로 빌드·실행해 API 호환(이름 변경 잦음)을 먼저 확인할 것.

> ⚠️ **Kotlin 플러그인을 별도로 적용하지 말 것**: AGP 9.0+ 는 built-in Kotlin support 가 기본
> 활성화(`android.builtInKotlin=true`)돼 있어 `org.jetbrains.kotlin.android`(`kotlin-android`)를
> `plugins {}` 에 같이 넣으면 `Cannot add extension with name 'kotlin'` 로 빌드가 깨진다.
> `kotlin { compilerOptions { ... } }` 블록은 built-in Kotlin이 자동 생성하는 extension이라
> 플러그인 없이도 그대로 쓸 수 있다. (AGP 10.0 에서는 opt-out인 `android.builtInKotlin=false` 자체가 사라짐.)

## 빌드 & 검증 순서
사전 조건(로컬): Android SDK + NDK, JDK 21, `local.properties` 또는 `ANDROID_HOME`.

```bash
# ⓪ C++ 코어 로직만 초고속 검증 — Android/SDK 불필요, 어느 환경에서든 가능
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  rag-library/src/main/cpp/vector_index.cpp \
  rag-library/src/main/cpp/tests/vector_index_test.cpp \
  -o /tmp/vector_index_test && /tmp/vector_index_test   # 기대 출력: ALL OK
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  rag-library/src/main/cpp/vector_index.cpp \
  rag-library/src/main/cpp/hnsw_index.cpp \
  rag-library/src/main/cpp/tests/hnsw_index_test.cpp \
  -o /tmp/hnsw_index_test && /tmp/hnsw_index_test       # 기대 출력: ALL OK (recall@10 포함)
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  rag-library/src/main/cpp/vector_index.cpp \
  rag-library/src/main/cpp/hnsw_index.cpp \
  rag-library/src/main/cpp/index_io.cpp \
  rag-library/src/main/cpp/tests/index_io_test.cpp \
  -o /tmp/index_io_test && /tmp/index_io_test           # 기대 출력: ALL OK

# ① C++17 컴파일 + CMake 링킹 확인 (librag-core.so 생성)
./gradlew :rag-library:externalNativeBuildDebug

# ② 호스트 유닛테스트 + AAR 산출
./gradlew :rag-library:testDebugUnitTest :rag-library:assembleRelease

# ③ 실기기/에뮬레이터 계측 테스트 — JNI 왕복 + 검색 E2E (핵심 게이트)
./gradlew :rag-library:connectedDebugAndroidTest
```

> ⚠️ **유닛테스트로는 JNI 검증 불가.** `./gradlew test`(host JVM)는 `.so`를 로드하지 못한다.
> `System.loadLibrary` 와 JNI 왕복은 반드시 ③(계측) 또는 CI instrumented 잡으로 확인한다.

### CI (GitHub Actions, `.github/workflows/ci.yml`)
`main`/`honey` push 및 PR마다 자동 실행:
- **build 잡**: ⓪ C++ 호스트 테스트 → ② 유닛테스트 + AAR (ubuntu, JDK 21)
- **instrumented 잡**: ③ 을 에뮬레이터(API 34, x86_64, KVM)에서 실행
→ **SDK 없는 환경(클라우드 세션)에서도 push 후 Actions 탭에서 초록불 확인 가능.**

## JNI 규칙 — 새 네이티브 함수 추가 시 필독
- **패키지/클래스**: Kotlin `com.example.rag_library`. 네이티브 클래스: `OnDeviceRAGCore`(스모크), `NativeVectorIndex`(인덱스), `NativeLlm`(LLM).
- **텍스트 전달 규약(NativeLlm)**: 프롬프트/생성 결과는 `ByteArray`(표준 UTF-8) 로 왕복 —
  `jstring`(Modified UTF-8)은 이모지 등 보충 평면에서 변형됨. 스트리밍 콜백은 C++ 쪽
  `completeUtf8Prefix` 가 문자 경계를 보장한 조각만 올린다.
- **심볼 이름**: `Java_<패키지경로(_구분)>_<클래스>_<메서드>`. 패키지명 안의 `_`는 **`_1`로 이스케이프**.
  - 예: `NativeVectorIndex#nativeAdd` → `Java_com_example_rag_1library_NativeVectorIndex_nativeAdd`
  - 한 글자라도 틀리면 런타임 `UnsatisfiedLinkError`. (함수가 더 늘면 `JNI_OnLoad` + `RegisterNatives` 검토.)
- **정적/인스턴스**: `@JvmStatic`(companion) → `(JNIEnv*, jclass)`, 인스턴스 → `(JNIEnv*, jobject)`.
  `NativeVectorIndex` 의 11개 함수(nativeCreate/nativeCreateHnsw/nativeDestroy/nativeClear/
  nativeAdd/nativeRemove/nativeSize/nativeSearch(allowMask 포함)/nativeSave/nativeLoad/nativeDim)는
  전부 `@JvmStatic` + 핸들(jlong) 전달.
  (함수가 더 늘면 `JNI_OnLoad` + `RegisterNatives` 전환 검토 — 지금이 임계점.)
- **필터 전달 규약**: docId 필터는 id 인덱스 ByteArray 마스크로 1회 전달 — 후보별 자바
  콜백 업콜 금지(µs 업콜이 청크 수만큼 터진다). C++ 스캔 루프 내부에서 검사한다.
- **라이브러리 이름 3중 일치**: `System.loadLibrary("rag-core")` ↔ `add_library(rag-core ...)` ↔ `librag-core.so`.
- **FloatArray 전달 패턴**(rag_jni.cpp 참고): `GetPrimitiveArrayCritical` → `memcpy` → 즉시 `Release(..., JNI_ABORT)`.
  Critical 구간 안에서 JNI 호출·힙 할당·블로킹 금지. 출력은 `Set{Int,Float}ArrayRegion` 으로 out-배열에 기록.
- **심볼 사전 검증(로컬/클라우드 공통)**: JDK 헤더로 컴파일 후 `nm` 확인 —
  `g++ -std=c++17 -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -fPIC -c rag_jni.cpp && nm -g rag_jni.o | grep Java_`

## 성능 (호스트 x86_64 벤치마크 — dim=384·무작위 데이터 = 최악 조건)
측정 도구: `tests/bench.cpp` (파일 상단 명령으로 수동 실행 — CI 러너 타이밍은 노이즈라 미포함).
- **dot 마이크로벤치**: 스칼라 43.2 → SIMD 10.6 ms (2천만 회, **4.1×**). `dot_product.h`
  (arm64=NEON·x86_64=SSE2).
- **규모 스윕**(브루트 검색 ms/질의): 1k=0.06 · 10k=0.62 · 20k=1.67 · 50k=7.06.
  HNSW 는 20k 에서 0.43 ms(ef=100). 브루트 전체 개선 8.79 → 1.68 ms(5.2×, SIMD+k-힙).
- 브루트포스 topK 는 k-힙 방식(후보 전체 배열 미생성, O(k) 메모리).
- **HNSW recall 은 hnswlib(레퍼런스)와 동률** — 동일 데이터·파라미터에서 ef 100/200/400/800
  스윕 전 구간 ±0.02 이내 (mine 0.352/0.565/0.779/0.923 vs hnswlib 0.365/0.582/0.793/0.935).
  무작위 고차원은 거리 집중으로 재현율이 낮게 나오는 최악 조건이며, 구조가 있는 실제
  임베딩(e5)에선 같은 ef 에서 훨씬 높다. 기본 efSearch=128.
- HNSW 이웃 선택은 Malkov Algorithm 4 다양성 휴리스틱(+keepPruned) 사용.
- **JNI 경계 정직성**: 질의 복사 1.5KB ≈ 무시 가능(검색 0.4~7ms 대비), C++ add 2~16µs/벡터
  — 인덱싱 병목은 임베딩 추론(ms/청크)이라 배치 JNI 미도입(측정 근거, README ④ 참고).
- 배포: `maven-publish` + `jitpack.yml` — 태그 push 후 JitPack 소비 가능(README '설치').
  제출물 준비 노트(보고서 문구·영상 대본): `docs/SUBMISSION.md`.

## 절대 커밋 금지
- 모델 가중치: `*.onnx`, `*.gguf`, `*.task`, `*.tflite`, `/models/` — 재배포 라이선스 문제 + GitHub 100MB 제한. 앱 최초 실행 시 다운로드하는 설계.
- `local.properties` (기기별 SDK 경로).

## 로드맵 & 현재 위치
- **Phase 0**: ✅ JNI 왕복 뼈대 ✅ C++ 코사인 top-k ✅ JNI 브리지 ✅ 청킹/해싱 임베더/RagEngine
  — 남은 것: 실기기/CI 에서 ③ 초록불 확인
- **Phase 0.5 (코드 완료)**: ✅ UnigramTokenizer(순수 Kotlin SentencePiece, Viterbi)
  ✅ OnnxEmbedder(ONNX Runtime 1.27.0 + e5 프리픽스) ✅ scripts/prepare_model.py
  — **남은 것**: 개발 PC에서 모델 준비 → adb push → OnnxEmbedderTest 실기기 검증,
  UnigramTokenizer 를 실제 e5 vocab 골든과 교차검증(`prepare_model.py --golden`)
- **Phase 1 ✅ (CI 초록불)**: HnswIndex(호스트 recall@10=1.000) + JNI + RagEngine IndexKind —
  에뮬레이터 계측 통과 확인됨.
- **Phase 2 (코드 완료)**: ✅ index_io 직렬화('RAG1' 포맷, mmap 친화 레이아웃)
  ✅ NativeVectorIndex.save/load ✅ RagEngine.save/load(청크 메타 포함 스냅샷)
  — 남은 것: CI/실기기 확인. mmap 제로카피 로드는 추후 최적화 항목.
- **Phase 3 (코드 완료)**: ✅ llama.cpp b9893 정적 링크 ✅ LlmEngine(스트리밍 generate,
  UTF-8 경계) ✅ RagChat + QwenPromptBuilder(ChatML) — 호스트 스모크(vocab GGUF) 통과.
  — **남은 것**: Qwen2.5-1.5B GGUF 를 기기에 배치해 LlmEngineTest 실기기 검증(README 참고).
- **Phase 4 (코드 완료)**: ✅ 데모 앱 UI (`:app` MainActivity) — 시연영상(3분)용.
  `./gradlew :app:installDebug` 로 설치. CI build 잡이 `:app:assembleDebug` 컴파일 검증.
  — 남은 것: 실기기에서 화면 확인 + 시연영상 촬영.
- ~~라이선스 정비~~ ✅ 완료: `LICENSE`(Apache-2.0) + `NOTICE` + `THIRD_PARTY_LICENSES.md` + `README.md`.
- **검색 CRUD ✅**: search(allowDocIds 필터) / removeDocument(브루트포스 swap-remove) /
  clear. RagChat.ask 는 onSources 를 토큰보다 먼저 콜백(체감 대기 제거).
- **결선 카드(제출 후, 데모 안정이 전제)**: ① SQ8 int8 스칼라 양자화 — 메모리 4배↓
  (5만 청크 75→19MB), NEON vmull_s8(+가능하면 SDOT 런타임 분기), recall 곡선 벤치 헤드라인.
  ② 구조 인지 청킹(마크다운/한국어 종결어미) ③ BM25(문자 bigram 자체 구현)+RRF 하이브리드.
  제외 확정: 크로스인코더 리랭커(메모리 예산 초과), QNN/Hexagon(독점 SDK — Apache-2.0 부적합),
  NNAPI(deprecated), Vulkan 오프로드(기기 편차 — 멘토링 단계 소재).

## 대회 일정 (2026, 핵심 마감)
- 참가접수: ~**7.17(금) 18:00**, oss.kr 온라인 접수. (최우선)
- 출품작 제출: **8.27(목) 18:00** — 결과보고서 · 시연영상(3분) · 소스코드.
- 1차 서면평가 9.3 / 시상식 12.4. (오리엔테이션 7.23, 멘토링·2차 평가는 결선 확정 후 포털 재확인)

## 이 원격(클라우드) 세션 주의
이 컨테이너에는 Android SDK/NDK가 없어 Gradle Android 빌드·계측 테스트를 **직접 실행할 수 없다**.
여기서 가능한 검증: ⓪ C++ 호스트 테스트(g++), JNI 심볼 nm 확인, jshell 로직 검증, 정적 검토.
①~③ 초록불은 **push 후 GitHub Actions** 또는 SDK가 있는 로컬(Android Studio / 데스크톱 Claude Code)에서 확인한다.
