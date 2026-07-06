#!/usr/bin/env python3
"""multilingual-e5-small 을 온디바이스 RAG 용으로 준비하는 스크립트 (개발 PC에서 실행).

하는 일:
  1) HuggingFace 모델을 ONNX 로 내보내기 (optimum, --skip-export 로 생략 가능)
  2) tokenizer.json → vocab.tsv 변환 (라이브러리의 UnigramTokenizer 입력 형식)
  3) (옵션) HF 토크나이저 골든 토큰화 출력 — UnigramTokenizer 교차검증용

사전 준비:
    pip install "optimum[exporters]"

사용:
    python scripts/prepare_model.py --out models
    python scripts/prepare_model.py --out models --golden "김치는 발효 음식이다"

기기 배치 (계측 테스트 OnnxEmbedderTest / 데모가 이 경로를 본다):
    adb shell mkdir -p /data/local/tmp/rag-models
    adb push models/model.onnx /data/local/tmp/rag-models/
    adb push models/vocab.tsv  /data/local/tmp/rag-models/

주의: models/ 는 .gitignore 에 있다 — 모델 산출물은 절대 커밋하지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-small"
HEADER = "#sp-unigram-v1"


def export_onnx(out: Path) -> None:
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        sys.exit('optimum 미설치 → pip install "optimum[exporters]"')
    print(f"ONNX 내보내기: {MODEL_ID} → {out}/ (수 분 소요)")
    main_export(MODEL_ID, output=str(out), task="feature-extraction")


def convert_tokenizer(tok_json: Path, tsv: Path) -> None:
    tj = json.loads(tok_json.read_text(encoding="utf-8"))
    model = tj["model"]
    if model.get("type") != "Unigram":
        sys.exit(f"Unigram 토크나이저가 아님: {model.get('type')!r}")
    vocab = model["vocab"]  # [[piece, score], ...] — 리스트 인덱스 == 토큰 id
    unk = model["unk_id"]
    ids = {piece: i for i, (piece, _score) in enumerate(vocab)}
    try:
        bos, eos = ids["<s>"], ids["</s>"]
    except KeyError as e:
        sys.exit(f"vocab 에 특수 토큰 없음: {e}")
    with tsv.open("w", encoding="utf-8", newline="\n") as w:
        w.write(f"{HEADER}\tunk={unk}\tbos={bos}\teos={eos}\n")
        for piece, score in vocab:
            if "\t" in piece or "\n" in piece or "\r" in piece:
                sys.exit(f"TSV 로 표현할 수 없는 piece: {piece!r}")
            w.write(f"{piece}\t{score}\n")
    print(f"vocab.tsv 생성: {tsv} (pieces={len(vocab)}, unk={unk}, bos={bos}, eos={eos})")


def print_golden(out: Path, text: str) -> None:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("골든 출력에는 tokenizers 필요 → pip install tokenizers")
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    enc = tok.encode(text)
    print(f"HF 골든 — {text!r}")
    print(f"  ids    = {list(enc.ids)}")
    print(f"  pieces = {list(enc.tokens)}")
    print("(UnigramTokenizer.encode 결과와 비교해 토크나이저를 교차검증할 것)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="models", type=Path, help="산출물 디렉터리 (기본: models)")
    ap.add_argument("--skip-export", action="store_true", help="ONNX 내보내기 생략(vocab 변환만)")
    ap.add_argument("--golden", metavar="TEXT", help="HF 토크나이저 골든 토큰화 출력")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_export:
        export_onnx(args.out)
    convert_tokenizer(args.out / "tokenizer.json", args.out / "vocab.tsv")
    if args.golden:
        print_golden(args.out, args.golden)
    print("\n다음 단계: adb push 로 model.onnx / vocab.tsv 를 /data/local/tmp/rag-models/ 에 배치")


if __name__ == "__main__":
    main()
