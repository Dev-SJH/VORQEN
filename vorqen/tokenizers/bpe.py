"""
BPE (Byte Pair Encoding) 토크나이저
- 자주 등장하는 글자 쌍을 반복적으로 병합
- GPT, Llama 등 현대 LLM의 표준 방식
- 밑바닥부터 직접 구현
"""
import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """BPE 토크나이저 (밑바닥 구현)"""

    def __init__(self, merges=None, vocab=None):
        # merges: [(a, b), ...] 병합 순서
        # vocab: {token_id: str}
        self.merges = merges or []
        self.vocab = vocab or {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size=1000, verbose=True):
        """텍스트에서 BPE 학습"""
        # 1. 초기: 모든 글자를 개별 토큰으로
        words = list(text)
        vocab_set = set(words)

        # 2. 초기 vocab (글자 → id)
        vocab = {}
        for ch in sorted(vocab_set):
            vocab[len(vocab)] = ch

        merges = []
        num_merges = vocab_size - len(vocab_set)

        if verbose:
            print(f"초기 어휘: {len(vocab_set)} 글자")
            print(f"목표 병합: {num_merges}회")

        # 3. 반복: 가장 자주 나오는 쌍 병합
        for i in range(num_merges):
            pairs = Counter(zip(words[:-1], words[1:]))
            if not pairs:
                break

            best_pair, count = pairs.most_common(1)[0]
            if count < 2:  # 두 번 이상 나온 쌍만 병합
                break

            # 병합 실행
            new_word = best_pair[0] + best_pair[1]
            new_words = []
            j = 0
            while j < len(words):
                if (
                    j < len(words) - 1
                    and words[j] == best_pair[0]
                    and words[j + 1] == best_pair[1]
                ):
                    new_words.append(new_word)
                    j += 2
                else:
                    new_words.append(words[j])
                    j += 1
            words = new_words

            merges.append(best_pair)
            vocab[len(vocab)] = new_word

            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}/{num_merges} | pair={best_pair} | vocab={len(vocab)}")

        if verbose:
            print(f"✅ 학습 완료 | 최종 어휘: {len(vocab)}")

        return cls(merges, vocab)

    def _apply_merges_to_word(self, chars):
        """한 단어(char 리스트)에 merges 순서대로 적용"""
        chars = list(chars)
        for a, b in self.merges:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(a + b)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars

    def encode(self, text):
        """텍스트 → 토큰 id 리스트"""
        # 글자 단위로 쪼갠 후 merges 적용
        tokens = self._apply_merges_to_word(list(text))

        # 토큰 → id
        token_to_id = {v: k for k, v in self.vocab.items()}
        ids = []
        for tok in tokens:
            if tok in token_to_id:
                ids.append(token_to_id[tok])
            else:
                # 모르는 글자는 각 글자로 쪼개서
                for ch in tok:
                    if ch in token_to_id:
                        ids.append(token_to_id[ch])
        return ids

    def decode(self, ids):
        """토큰 id 리스트 → 텍스트"""
        return "".join(self.vocab[i] for i in ids if i in self.vocab)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON은 키가 str이어야 함
        data = {
            "merges": self.merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
def train(cls, text, vocab_size=1000, min_freq=5, verbose=True):
    """텍스트에서 BPE 학습 (드문 글자 필터링 포함)"""
    from collections import Counter

    # 1. 드문 글자 필터링 (한국어는 글자가 너무 많음)
    char_counts = Counter(text)
    valid_chars = {c for c, n in char_counts.items() if n >= min_freq}
    filtered = "".join(c for c in text if c in valid_chars)

    if verbose:
        print(f"원본 글자 종류: {len(char_counts)}")
        print(f"필터 후 (≥{min_freq}회): {len(valid_chars)}")

    # 2. 초기 vocab
    vocab_set = set(filtered)
    vocab = {i: ch for i, ch in enumerate(sorted(vocab_set))}
    merges = []
    num_merges = vocab_size - len(vocab_set)

    if verbose:
        print(f"초기 어휘: {len(vocab_set)}")
        print(f"목표 병합: {num_merges}회")

    if num_merges <= 0:
        if verbose:
            print("⚠️  vocab_size가 너무 작음. 병합 건너뜀.")
        return cls(merges, vocab)

    # 3. BPE 병합
    words = list(filtered)
    for i in range(num_merges):
        pairs = Counter(zip(words[:-1], words[1:]))
        if not pairs:
            break
        best_pair, count = pairs.most_common(1)[0]
        if count < 2:
            break

        new_word = best_pair[0] + best_pair[1]
        new_words = []
        j = 0
        while j < len(words):
            if (j < len(words) - 1
                    and words[j] == best_pair[0]
                    and words[j + 1] == best_pair[1]):
                new_words.append(new_word)
                j += 2
            else:
                new_words.append(words[j])
                j += 1
        words = new_words

        merges.append(best_pair)
        vocab[len(vocab)] = new_word

        if verbose and (i + 1) % 100 == 0:
            print(f"  merge {i+1}/{num_merges} | vocab={len(vocab)}")

    if verbose:
        print(f"✅ 학습 완료 | 최종 어휘: {len(vocab)}")

    return cls(merges, vocab)