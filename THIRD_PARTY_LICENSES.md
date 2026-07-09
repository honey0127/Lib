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

## AI 모델 활용 명세 (오픈소스 개발자대회 운영규정 제9조 대응)

두 모델 모두 **외부 공개 모델을 추가 학습(파인튜닝) 없이 그대로 활용**하는
유형(운영규정 제9조 제2항 제1호)이며, 가중치가 공개 저장소(Hugging Face)에서
접근 제한 없이 무상 공개된 **오픈웨이트 이상** 모델이다(제9조 제1항 충족).

| 항목 | 임베딩 모델 | 답변 생성 모델 |
|---|---|---|
| 모델명 | intfloat/multilingual-e5-small | Qwen/Qwen2.5-1.5B-Instruct (GGUF) |
| 라이선스 | MIT | Apache-2.0 |
| 공개 수준 | 오픈웨이트(가중치 공개) | 오픈웨이트(가중치 공개) |
| 활용 방식 | 그대로 활용 — ONNX 변환(optimum) 후 온디바이스 추론 | 그대로 활용 — 공식 GGUF 양자화본 온디바이스 추론 |
| 추론 코드 | 본 저장소 직접 작성(Apache-2.0): `OnnxEmbedder`, `UnigramTokenizer` | 본 저장소 직접 작성(Apache-2.0): `LlmEngine` + llama.cpp(MIT) 정적 링크 |
| 외부 API 의존 | **없음 — 완전 온디바이스**(추론 시 네트워크 0회) | **없음 — 완전 온디바이스**(추론 시 네트워크 0회) |

- 참가팀이 직접 작성한 모든 소스코드(추론 코드 포함)에는 OSI 인증 라이선스인
  **Apache-2.0** 을 적용한다(운영규정 제8조).
- 참고: HNSW 재현율 검증에는 hnswlib(Apache-2.0)를 **비교 실험에만** 사용했으며,
  해당 코드는 본 저장소에 포함되지 않는다(구현은 원논문 기준 자체 작성).
- 개발 과정에서 코드 작성 보조 도구로 상용 AI 서비스(Claude)를 활용했으며(운영규정
  제9조 제5항 허용 사항), 커밋 이력에 `Co-Authored-By` 로 투명하게 고지한다.
