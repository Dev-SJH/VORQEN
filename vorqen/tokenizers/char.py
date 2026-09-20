"""
글자 단위 토크나이저
- 텍스트를 글자 하나하나로 쪼개서 숫자로 변환
- 가장 단순한 방식 (Phase 1~2용)
"""
import json
from pathlib import Path


class CharTokenizer:
    """글자 단위 토크나이저"""

    def __init__(self, chars=None):
        self.chars = chars or []
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_text(cls, text):
        """텍스트에서 등장하는 모든 글자 추출"""
        chars = sorted(list(set(text)))
        return cls(chars)

    def encode(self, text):
        """텍스트 → 숫자 리스트"""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        """숫자 리스트 → 텍스트"""
        return "".join([self.itos[i] for i in ids])

    def save(self, path):
        """토크나이저 저장 (JSON)"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """토크나이저 불러오기"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])