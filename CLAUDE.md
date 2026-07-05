# CLAUDE.md — 온디바이스 RAG 라이브러리

Android 앱에 **완전 오프라인** AI 문서 검색을 넣는 오픈소스 Kotlin 라이브러리. 성능 핵심 경로는 C++/JNI.
(2026 오픈소스 개발자대회 자유과제 · AI·보안/안전 트랙 출품작)

이 문서는 Claude Code / 개발자를 위한 **빌드·검증 명령 및 JNI 규칙의 단일 진실 소스**다.
반복 빌드-수정 루프에서 먼저 참조한다.

## 모듈 구조
- `:rag-library` — 라이브러리 본체(AAR). Kotlin API + `src/main/cpp` 의 C++/JNI 코어(`librag-core.so`). **개발의 중심.**
- `:app` — 데모/수동 테스트용 앱 셸(namespace `com.example.lib`). 현재 UI 없음(res+manifest만). 데모 UI는 Phase 4에서 채운다.

```
rag-library/src/main/
├── AndroidManifest.xml
├── cpp/
│   ├── CMakeLists.txt        # add_library(rag-core SHARED native-lib.cpp)
│   └── native-lib.cpp        # JNI 진입점
└── java/com/example/rag_library/
    └── OnDeviceRAGCore.kt     # loadLibrary("rag-core") + external fun stringFromJNI()
rag-library/src/androidTest/java/com/example/rag_library/
└── OnDeviceRAGCoreTest.kt     # JNI 왕복 검증(계측 테스트)
```

## 툴체인 (repo 실제 고정값)
| 항목 | 값 | 위치 |
|---|---|---|
| AGP | 9.2.1 | `gradle/libs.versions.toml` |
| Gradle | 9.4.1 | `gradle/wrapper/gradle-wrapper.properties` |
| Kotlin | 2.1.0 (잠정) | `gradle/libs.versions.toml` |
| compileSdk | 36 (minorApiLevel 1) | 각 모듈 `build.gradle.kts` |
| minSdk | 26 | |
| Java | 11 | `compileOptions` / `kotlinOptions.jvmTarget` |
| CMake | 3.22.1 | |
| NDK | SDK 기본값(미고정) | 재현성 위해 `ndkVersion` 고정 권장 |
| C++ | 17 | `cppFlags` / `CMakeLists.txt` |
| ABI | arm64-v8a, x86_64 | `rag-library` `defaultConfig.ndk.abiFilters` |

> ⚠️ **Kotlin 버전 주의**: 2.1.0 으로 잠정 고정했다. Gradle이 AGP 9.2.1 과의 호환성 오류를 내면
> Android Studio 가 번들한 Kotlin 버전에 맞춰 `libs.versions.toml` 의 `kotlin` 값을 조정할 것.
> (에러 메시지에 요구 버전이 그대로 찍힌다 → 한 줄 수정으로 해결.)

## 빌드 & 검증 — Phase 0 순서
사전 조건: Android SDK + NDK 설치, `local.properties` 의 `sdk.dir=` 또는 `ANDROID_HOME` 설정.

```bash
# ① C++17 컴파일 + CMake 링킹 확인 (librag-core.so 생성)
./gradlew :rag-library:externalNativeBuildDebug

# ② AAR 산출 (라이브러리 전체 컴파일 — Kotlin 포함)
./gradlew :rag-library:assembleRelease

# ③ 실기기/에뮬레이터에서 JNI 왕복 검증 (핵심 게이트, 기기 연결 필요)
./gradlew :rag-library:connectedDebugAndroidTest
```

③의 `OnDeviceRAGCoreTest.jniRoundTrip` 이 통과하면 Phase 0 위험 구간을 넘긴 것.

> ⚠️ **유닛테스트로는 JNI 검증 불가.** `./gradlew test`(host JVM)는 `.so`를 로드하지 못한다.
> `System.loadLibrary` 가 실제로 뜨는지는 반드시 ③(계측 테스트)로만 확인된다.
> 에뮬레이터로 돌린다면 ABI에 `x86_64` 가 포함돼 있어야 한다(이미 포함).

## JNI 규칙 — 새 네이티브 함수 추가 시 필독
- **패키지/클래스**: Kotlin `com.example.rag_library`, 네이티브 클래스 `OnDeviceRAGCore`.
- **심볼 이름**: `Java_<패키지경로(_구분)>_<클래스>_<메서드>`. 패키지명 안의 `_`는 **`_1`로 이스케이프**된다.
  - `com.example.rag_library.OnDeviceRAGCore#foo` → `Java_com_example_rag_1library_OnDeviceRAGCore_foo`
  - 한 글자라도 틀리면 런타임 `UnsatisfiedLinkError`. (함수 수가 늘면 `JNI_OnLoad` + `RegisterNatives` 패턴 검토.)
- **라이브러리 이름 3중 일치**: `System.loadLibrary("rag-core")` ↔ CMake `add_library(rag-core ...)` ↔ 산출물 `librag-core.so`.
- **시그니처**: `external fun` 파라미터 ↔ C++ 인자. 인스턴스 메서드면 `(JNIEnv*, jobject)`, `@JvmStatic`/companion 정적이면 `(JNIEnv*, jclass)`.

## 절대 커밋 금지
- 모델 가중치: `*.onnx`, `*.gguf`, `*.task`, `*.tflite`, `/models/` — 재배포 라이선스 문제 + GitHub 100MB 제한. 앱 최초 실행 시 다운로드하는 설계.
- `local.properties` (기기별 SDK 경로).

## 로드맵 (요약)
- **Phase 0 ← 현재**: JNI 왕복 초록불 + 얇은 뼈대. HNSW·mmap·배열 최적화는 아직 넣지 않는다.
- **Phase 0 다음 슬라이스**: ONNX 임베딩 + Kotlin 브루트포스 코사인 top-k("검색"만, LLM 없음). `FloatArray`를 C++로 넘기기 시작할 때 `GetPrimitiveArrayCritical` / Direct `ByteBuffer` 최적화를 설계.
- **Phase 1~3**: 벡터 인덱스(HNSW) · mmap 영속화 · LLM(GGUF) 통합.
- **Phase 4**: 데모 앱 UI (`:app`).
- **라이선스 정비(심사 항목)**: 루트 `LICENSE`(Apache-2.0 예정) + `THIRD_PARTY_LICENSES.md`(모델/런타임 출처·라이선스). 아직 미생성.

## 대회 일정 (2026, 핵심 마감)
- 참가접수: ~**7.17(금) 18:00**, oss.kr 온라인 접수. (최우선)
- 출품작 제출: **8.27(목) 18:00** — 결과보고서 · 시연영상(3분) · 소스코드.
- 1차 서면평가 9.3 / 시상식 12.4. (오리엔테이션 7.23, 멘토링·2차 평가는 결선 확정 후 포털 재확인)

## 이 원격(클라우드) 세션 주의
이 컨테이너에는 Android SDK/NDK가 없어 Gradle Android 빌드·계측 테스트를 **실행할 수 없다**.
여기서는 소스/설정 수정과 정적 검토까지만 가능하며, 실제 빌드·초록불 확인은 SDK가 설치된 로컬(Android Studio / 데스크톱 Claude Code)에서 수행한다.
