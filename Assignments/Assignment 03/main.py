import os

import torch
from transformers import AutoTokenizer

from model_solution import ModelConfig, Transformer


def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[Transformer, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_config = checkpoint["model_config"]
    if isinstance(raw_config, ModelConfig):
        config = raw_config
    else:
        config = ModelConfig(**raw_config)

    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main():
    checkpoint_path = "./checkpoints/tiny_model.pt"

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Mac MPS")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run training first."
        )

    model, _ = load_checkpoint(checkpoint_path, device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Once upon a time, "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids, num_new_tokens=30)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Prompt:", prompt)
    print("Generated:", generated_text)


if __name__ == "__main__":
    main()
