"""A simple training loop for our transformer model"""

import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from model_solution import ModelConfig, Transformer

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using Mac MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


def get_chunked_tinystories(
    chunk_size: int,
) -> Int[Tensor, "num_chunks chunk_size"]:

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load tiny stories dataset
    train_dataset = load_dataset("roneneldan/TinyStories")["train"]

    # We'll just grab the first 1%
    train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.01)))

    # Tokenize the dataset
    chunks: list[list[int]] = []
    current_chunk: list[int] = []
    for row in tqdm(train_dataset, desc="Tokenizing dataset"):
        document: str = row["text"]
        tokens: list[int] = tokenizer(
            document, truncation=True, max_length=chunk_size
        ).input_ids

        # Fill current chunk up to chunk_size
        current_chunk.extend(tokens)
        if len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size])
            # Reset the current chunk
            current_chunk = current_chunk[chunk_size:]

    # Sanity checks
    assert all(len(chunk) == chunk_size for chunk in chunks)

    return torch.tensor(chunks, dtype=torch.long)


def plot_results(
    losses: list[float],
    grad_norms: list[float],
    save_path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel - Loss curve
    ax1.plot(losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Right panel - Gradient norm
    ax2.plot(grad_norms)
    ax2.set_title("Gradient Norm")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Grad Norm")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    step: int,
    losses: list[float],
    grad_norms: list[float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "step": step,
            "losses": losses,
            "grad_norms": grad_norms,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model_config: ModelConfig,
) -> tuple[Transformer, torch.optim.Optimizer, int, list[float], list[float]]:
    checkpoint = torch.load(path, map_location=device)

    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=checkpoint.get("learning_rate", 1e-5)
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = int(checkpoint.get("step", 0))
    losses = list(checkpoint.get("losses", []))
    grad_norms = list(checkpoint.get("grad_norms", []))

    return model, optimizer, step, losses, grad_norms


def train(
    learning_rate: float,
    gradient_clipping: float | None,
    model_config: ModelConfig,
    batch_size: int,
    max_steps: int | None = None,
    checkpoint_path: str = "./checkpoints/model_checkpoint.pt",
    resume_from_checkpoint: str | None = None,
) -> None:

    if gradient_clipping is None:
        # This lets us just get the grad norm but we don't clip
        gradient_clipping = float("inf")

    chunk_size: int = model_config.context_length
    cached_dataset_path: str = (
        f"./datasets/tinystories_10pct_chunk_size_{chunk_size}.pt"
    )
    os.makedirs(os.path.dirname(cached_dataset_path), exist_ok=True)

    if os.path.exists(cached_dataset_path):
        dataset = torch.load(cached_dataset_path)
    else:
        dataset: Int[Tensor, "num_chunks chunk_size"] = get_chunked_tinystories(
            chunk_size
        )
        torch.save(dataset, cached_dataset_path)

    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model, optimizer, num_steps_completed, losses, grad_norms = load_checkpoint(
            resume_from_checkpoint, model_config
        )
        model = model.to(device)
    else:
        model = Transformer(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        num_steps_completed = 0
        losses = []
        grad_norms = []

    num_chunks: int = dataset.shape[0]

    if max_steps is not None:
        tqdm_max_steps = min(max_steps, num_chunks // batch_size)
    else:
        tqdm_max_steps = num_chunks // batch_size

    for i in tqdm(
        range(0, num_chunks, batch_size), desc="Training", total=tqdm_max_steps
    ):
        if max_steps is not None and num_steps_completed >= max_steps:
            break

        if num_steps_completed % 10 == 0 and num_steps_completed > 0:
            plot_results(losses, grad_norms, save_path="./losses_and_grad_norms.png")

        batch: Int[Tensor, "batch_size chunk_size"] = dataset[i : i + batch_size].to(
            device
        )

        optimizer.zero_grad()

        # Forward pass
        loss = model.get_loss_on_batch(batch)

        # Backward pass
        loss.backward()

        # Clip gradients
        with torch.no_grad():
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clipping
            ).item()
            grad_norms.append(grad_norm)

        optimizer.step()

        losses.append(loss.item())

        num_steps_completed += 1

    # Done with training, plot results in single plot
    plot_results(losses, grad_norms, save_path="./losses_and_grad_norms.png")

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        step=num_steps_completed,
        losses=losses,
        grad_norms=grad_norms,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Final loss after {num_steps_completed} steps: {losses[-1]:.6f}")
    print(f"Best loss during training: {min(losses):.6f}")
    print(f"Final gradient norm: {grad_norms[-1]:.6f}")


if __name__ == "__main__":
    tiny_model_config = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=3,
        context_length=128,
        vocab_size=50257,
    )

    train(
        learning_rate=1e-3,
        gradient_clipping=1,
        model_config=tiny_model_config,
        batch_size=16,
        max_steps=100,
        checkpoint_path="./checkpoints/tiny_model.pt",
        resume_from_checkpoint=None,
    )
