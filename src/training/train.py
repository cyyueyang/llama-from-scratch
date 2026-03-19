import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import time
from typing import Dict, Optional

from src.model.llama import LLaMA
from src.dataset.tokenizer import Tokenizer
from src.config.LLaMAConfig import LLaMAConfig
from src.dataset.dataset import TinyStoriesGPT2Dataset

class Trainer:
    def __init__(
            self,
            config: LLaMAConfig,
            model: LLaMA,
            tokenizer: Tokenizer,
            train_dataset: TinyStoriesGPT2Dataset,
            val_dataset: TinyStoriesGPT2Dataset,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model.to(self.config.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, betas=(0.9, 0.95))

        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = self.config.warmup_steps

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, warmup_steps, total_steps)

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = self.config.checkpoint_dir
        self.log_interval = self.config.log_interval

    def save_checkpoint(self, filename: str, is_best: bool):

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(checkpoint, best_path)
            print(f"best model saved to {best_path}")

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {self.current_epoch}")

        for step, batch in enumerate(pbar):
            x, y = batch
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            logits, loss = self.model(x, y)

            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            # 清空梯度
            self.optimizer.zero_grad()

            total_loss += loss.item()
            self.global_step += 1

            pbar.set_description(f"loss: {loss.item()}, lr: {self.scheduler.get_lr()}")

            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                print(f"global_step: {self.global_step}, avg_loss: {avg_loss}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"validating")

        for batch in pbar:
            x, y = batch
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            logits, loss = self.model(x, y)
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        print("="*50)
        print(f"Start training")
        print(f"Epoch : {self.config.epochs}")
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch : {epoch}, train_loss: {train_loss}")
            val_loss = self.validate()
            print(f"Epoch : {epoch}, val_loss: {val_loss}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"checkpoint_{epoch}.pth", True)
                print("Best model saved!!!!")
            else:
                self.save_checkpoint(f"checkpoint_{epoch}.pth", False)
        print("="*50)
        print(f"Finish training")

def main():
    config = LLaMAConfig()

    model = LLaMA(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")

    tokenizer = Tokenizer(config.vocab_path, config.merges_path)
    train_dataset = TinyStoriesGPT2Dataset(config, mode="train")
    val_dataset = TinyStoriesGPT2Dataset(config, mode="val")
    trainer = Trainer(config, model, tokenizer, train_dataset, val_dataset)

    trainer.train()

if __name__ == "__main__":
    main()



