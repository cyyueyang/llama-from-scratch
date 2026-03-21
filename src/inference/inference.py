import os
import torch

from src.model.llama import LLaMA
from src.dataset.tokenizer import Tokenizer
from src.config.LLaMAConfig import LLaMAConfig

def main():
    config = LLaMAConfig()
    checkpoint_path = os.path.join(config.checkpoint_dir, "model_best.pth")

    checkpoint = torch.load(checkpoint_path)

    tokenizer = Tokenizer(
        vocab_path=config.vocab_path,
        merges_path=config.merges_path,
    )

    model = LLaMA(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    prompts = [
        "Once upon a time",
        "In a small village",
        "The little rabbit",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("=" * 50)

        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=config.device)
        output_ids = model.generate(input_ids, max_new_token=256, temperature=1.0, top_k = 10, top_p=0.9, eos_token_id=tokenizer.vocab.get("<|endoftext|>", None))
        generated_text = tokenizer.decode(output_ids.tolist()[0])

        print(f"Generated text: {generated_text}")
        print("=" * 50)

if __name__ == "__main__":
    main()


