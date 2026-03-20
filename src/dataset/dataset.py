import torch
from torch.utils.data import Dataset
from src.dataset.tokenizer import Tokenizer
from src.config.LLaMAConfig import LLaMAConfig

class TinyStoriesGPT2Dataset(Dataset):
    def __init__(self, config: LLaMAConfig, mode = "train"):
        super().__init__()
        self.tokenizer = Tokenizer(vocab_path=config.vocab_path, merges_path=config.merges_path)
        self.config = config
        self.file_path: str = self.config.file_train_path if mode == "train" else self.config.file_val_path
        self.max_length: int = self.config.Dataset_max_len
        self.stride: int = self.config.Dataset_stride

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        raw_stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
        print(f"Loaded {len(raw_stories)} stories.")

        self.samples = []
        total_tokens = 0

        for story in raw_stories:

            encoding = self.tokenizer.encode(story + "<|endoftext|>")
            total_tokens += len(encoding)

            if len(encoding) < self.max_length:
                self._add_sample(encoding)
            else:
                for i in range(0, len(encoding), self.stride):
                    chunk = encoding[i: i + self.max_length]
                    if len(chunk) < 10:
                        continue
                    self._add_sample(chunk)

    def _add_sample(self, token_ids):
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.tokenizer.vocab["<|endoftext|>"]] * (self.max_length - len(token_ids))

        token_ids = token_ids[:self.max_length]
        self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.long)

        return x[:-1], x[1:]








