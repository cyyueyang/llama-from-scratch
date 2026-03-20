import json
import regex as re
from typing import Dict, List, Tuple

class Tokenizer:
    def __init__(self, vocab_path: str = r"./vocab.json", merges_path: str = r"./merges.txt") -> None:
        self.vocab_path = vocab_path
        self.merges_path = merges_path

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab: Dict[str, int] = json.load(f)

        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self.merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1: -1]:
                if line.strip():
                    a, b = line.strip().split()
                    self.merges.append((a, b))

        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self) -> Dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for i in range(256):
            if i not in bs:
                bs.append(i)
                cs.append(256 + i)
                n += 1

        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))

    def _get_pairs(self, word: Tuple[str, ...]) -> set:

        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char

        return pairs

    def _bpe(self, token: str) -> str:
        if token in self.vocab:
            return token

        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.merge_rank.get(pair, float("inf")))
            if bigram not in self.merge_rank:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2

                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break

            pairs = self._get_pairs(word)

        return " ".join(word)

    def encode(self, text: str) -> list[int]:

        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokens = []
        for token in re.findall(pattern, text):

            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_result = self._bpe(token_translated)

            for bpe_token in bpe_result.split(' '):

                tokens.append(self.vocab.get(bpe_token, self.vocab.get(" ", 0)))

        return tokens

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[id] for id in ids]
        text = "".join(tokens)
        bytes_list = [self.byte_decoder[c] for c in text]
        return bytes(bytes_list).decode("utf-8", errors="replace")


if __name__ == "__main__":
    # 测试用例
    tokenizer = Tokenizer()
    print(len(tokenizer.vocab))
    # 测试 1：基本编码解码
    text = "_"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"

    # 测试 2：包含空格的处理（GPT-2 中空格会变成 Ġ）
    text2 = " hello world"  # 前导空格
    ids2 = tokenizer.encode(text2)
    print(f"Encoded '{text2}': {ids2}")

    # 测试 3：特殊字符
    text3 = "GPT-2's tokenizer"
    ids3 = tokenizer.encode(text3)
    print(f"Tokens: {[tokenizer.id_to_token[i] for i in ids3]}")

    print("All tests passed!")



