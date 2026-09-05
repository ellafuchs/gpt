import torch
import argparse

from model import GPTLanguageModel


# -----------------------------
# Training utilities
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def pick_device(force_device: str | None = None):
    if force_device:
        return force_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny GPT on a text file (char-level).")

    p.add_argument("--file", default="input.txt", help="Path to input text file.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"],
                   help="Force device. Default: auto (cuda->mps->cpu).")
    p.add_argument("--gen-tokens", type=int, default=500, help="How many tokens to generate at the end.")

    return p.parse_args()

def main():
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)

    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters | device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    from tqdm import trange
    pbar = trange(args.max_iters, desc="training", leave=True)

    for it in pbar:
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model)
            pbar.set_postfix(
                train=f"{losses['train']:.4f}",
                val=f"{losses['val']:.4f}",
            )
        

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(step_loss=f"{loss.item():.4f}", refresh=False)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.gen_tokens)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()