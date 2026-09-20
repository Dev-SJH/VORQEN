"""
VORQEN CLI 진입점
사용법:
  python -m vorqen.cli.main train --config configs/model/tiny.yaml
  python -m vorqen.cli.main generate --config configs/model/tiny.yaml --prompt "안녕"
"""
import argparse
import sys
from pathlib import Path

import torch

from ..models.transformer import VORQEN
from ..tokenizers.char import CharTokenizer
from ..training.trainer import Trainer
from ..utils.config import load_config


def cmd_train(args):
    print(f"📂 설정 로드: {args.config}")
    cfg = load_config(args.config)
    print(f"   모델: {cfg.get('name', 'unnamed')}")

    # 데이터 로드
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 데이터 파일 없음: {data_path}")
        print("   먼저 data/raw/corpus.txt 파일을 준비하세요.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"📖 데이터: {len(text):,} 글자")

    # 토크나이저
    tokenizer = CharTokenizer.from_text(text)
    print(f"🔤 어휘 크기: {tokenizer.vocab_size}")

    # 텐서 변환
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"   train: {len(train_data):,} / val: {len(val_data):,}")

    # 디바이스
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  device: {device}")

    # 학습
    trainer = Trainer(cfg, tokenizer, train_data, val_data, device=device)
    trainer.train()

    # 저장
    save_path = Path(args.output) / cfg.get("name", "vorqen")
    trainer.save(save_path / "model.pt")

    # 생성 미리보기
    print("\n=== 생성 샘플 ===")
    tokenizer_ref = trainer.tokenizer
    for prompt in ["안녕", "대한민국"]:
        ids = tokenizer_ref.encode(prompt)
        if not ids:
            continue
        ctx = torch.tensor([ids], dtype=torch.long, device=device)
        out = trainer.model.generate(ctx, max_new_tokens=100, temperature=0.8, top_k=20)
        print(f"\n[입력: {prompt}]")
        print(tokenizer_ref.decode(out[0].tolist()))


def cmd_generate(args):
    cfg = load_config(args.config)
    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location=args.device)
    tokenizer = CharTokenizer.load(model_dir / "tokenizer.json")

    model = VORQEN(
        vocab_size=ckpt["vocab_size"],
        **{k: v for k, v in ckpt["config"]["model"].items() if k != "vocab_size"},
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device).eval()

    ids = tokenizer.encode(args.prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = model.generate(
        ctx, max_new_tokens=args.max_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


def main():
    parser = argparse.ArgumentParser(prog="vorqen", description="VORQEN CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="모델 학습")
    p_train.add_argument("--config", default="configs/model/tiny.yaml")
    p_train.add_argument("--data", default="data/raw/corpus.txt")
    p_train.add_argument("--output", default="checkpoints")
    p_train.add_argument("--device", default="auto")
    p_train.set_defaults(func=cmd_train)

    # generate
    p_gen = sub.add_parser("generate", help="텍스트 생성")
    p_gen.add_argument("--config", default="configs/model/tiny.yaml")
    p_gen.add_argument("--model_dir", default="checkpoints/vorqen-tiny")
    p_gen.add_argument("--prompt", default="안녕")
    p_gen.add_argument("--max_tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top_k", type=int, default=20)
    p_gen.add_argument("--device", default="cpu")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()