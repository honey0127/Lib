# Third-Party Licenses

이 프로젝트가 사용하는(또는 사용 예정인) 서드파티 구성요소와 라이선스 목록.

## 현재 사용 중

| 구성요소 | 라이선스 | 출처 | 용도 |
|---|---|---|---|
| ONNX Runtime (onnxruntime-android) | MIT | https://github.com/microsoft/onnxruntime | 임베딩 모델 추론 |
| llama.cpp (태그 b9893, CMake FetchContent 정적 링크) | MIT | https://github.com/ggml-org/llama.cpp | LLM(GGUF) 온디바이스 추론 |
| intfloat/multilingual-e5-small | MIT | https://huggingface.co/intfloat/multilingual-e5-small | 다국어 임베딩 모델 (기기에서 다운로드/배치, 커밋 안 함) |
| Qwen2.5-1.5B-Instruct (GGUF) | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF | 답변 생성 모델 (기기에서 다운로드/배치, 커밋 안 함) |
| AndroidX (core-ktx, appcompat, test) | Apache-2.0 | https://github.com/androidx/androidx | 라이브러리/테스트 |
| Material Components for Android | Apache-2.0 | https://github.com/material-components/material-components-android | 데모 UI |
| JUnit 4 (테스트 전용) | EPL-1.0 | https://github.com/junit-team/junit4 | 테스트 |

> **모델 가중치는 이 저장소에 포함하지 않는다** (`.gitignore`로 차단: `*.onnx`, `*.gguf`, `/models/` 등).
> 앱 최초 실행 시 위 출처에서 다운로드하는 설계이며, 각 구성요소의 라이선스 전문은 해당 출처를 따른다.
