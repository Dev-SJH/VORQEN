"""
학습 루프
"""
import time
from pathlib import Path

import torch

from ..data.dataset import get_batch
from ..models.transformer import VORQEN
from ..tokenizers.char import CharTokenizer


class Trainer:
    """VORQEN 학습 담당"""

    def __init__(self, cfg, tokenizer, train_data, val_data, device="cpu"):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

        m = cfg["model"]
        self.block_size = m["block_size"]

        # 모델 생성
        self.model = VORQEN(
            vocab_size=tokenizer.vocab_size,
            block_size=m["block_size"],
            n_embd=m["n_embd"],
            n_head=m["n_head"],
            n_layer=m["n_layer"],
            dropout=m["dropout"],
        ).to(device)

        # 옵티마이저
        t = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t["lr"],
            weight_decay=t.get("weight_decay", 0.1),
        )
        self.batch_size = t["batch_size"]
        self.max_steps = t["max_steps"]
        self.eval_interval = t.get("eval_interval", 500)
        self.eval_iters = t.get("eval_iters", 100)
        self.grad_clip = t.get("grad_clip", 1.0)

    @torch.no_grad()
    def estimate_loss(self):
        """검증 손실 추정"""
        self.model.eval()
        losses = {}
        for split, data in [("train", self.train_data), ("val", self.val_data)]:
            total = 0.0
            for _ in range(self.eval_iters):
                x, y = get_batch(data, self.block_size, self.batch_size, self.device)
                _, loss = self.model(x, y)
                total += loss.item()
            losses[split] = total / self.eval_iters
        self.model.train()
        return losses

    def train(self):
        """학습 실행"""
        print(f"🚀 학습 시작 | 파라미터: {self.model.num_params() / 1e6:.3f}M")
        print(f"   device={self.device} | block_size={self.block_size}")
        print(f"   batch_size={self.batch_size} | max_steps={self.max_steps}")
        print("-" * 60)

        self.model.train()
        t0 = time.time()

        for step in range(self.max_steps):
            x, y = get_batch(
                self.train_data, self.block_size, self.batch_size, self.device
            )
            _, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"step {step:5d} | loss {loss.item():.4f} | {elapsed:.1f}s")

            if step > 0 and step % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"  📊 eval @ {step} | "
                    f"train {losses['train']:.4f} | val {losses['val']:.4f}"
                )

        print("-" * 60)
        print(f"✅ 학습 완료 | 총 {time.time() - t0:.1f}초")
        return self.model

    def save(self, path):
        """모델 + 토크나이저 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.cfg,
                "vocab_size": self.tokenizer.vocab_size,
            },
            path,
        )
        self.tokenizer.save(path.parent / "tokenizer.json")
        print(f"💾 저장 완료: {path}")