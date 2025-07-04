import re
import sys
import numpy as np

def moving_avg(arr, window):
    if len(arr) < window:
        return []
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def parse_log_into_chunks(lines):
    chunks = []
    grad_norms = []
    for line in lines:
        epoch_match = re.search(r"epoch[: ]+(\d+)", line, re.I)
        t_loss = re.search(r"train.*loss[: =]+([0-9.]+)", line, re.I)
        v_loss = re.search(r"val.*loss[: =]+([0-9.]+)", line, re.I)
        acc = re.search(r"acc(?:uracy)?[: =]+([0-9.]+)", line, re.I)
        lr_match = re.search(r"lr[: =]+([0-9.eE-]+)", line)
        grad_norm_match = re.search(r"grad(?:ient)?[_ ]*norm[: =]+([0-9.eE+-]+)", line, re.I)
        nan_or_inf = bool(re.search(r"nan|inf", line, re.I))
        if epoch_match:
            chunk = {
                "epoch": int(epoch_match.group(1)),
                "train_loss": float(t_loss.group(1)) if t_loss else None,
                "val_loss": float(v_loss.group(1)) if v_loss else None,
                "acc": float(acc.group(1)) if acc else None,
                "lr": float(lr_match.group(1)) if lr_match else None,
                "grad_norm": float(grad_norm_match.group(1)) if grad_norm_match else None,
                "nan_or_inf": nan_or_inf,
                "line": line.strip(),
                "tags": [],
                "explanations": [],
            }
            chunks.append(chunk)
            if grad_norm_match:
                grad_norms.append(float(grad_norm_match.group(1)))
    return chunks, grad_norms

def analyze_chunks(chunks, grad_norms, num_classes=2):
    median_grad = np.median(grad_norms) if grad_norms else None
    for i, chunk in enumerate(chunks):
        if chunk['nan_or_inf']:
            chunk['tags'].append("instability")
            chunk['explanations'].append("NaN/Inf detected in metrics; likely too high LR or unstable model.")
        if chunk.get("grad_norm") is not None:
            gn = chunk["grad_norm"]
            if gn > 10 * (median_grad or 1):
                chunk['tags'].append("exploding_gradient")
                chunk['explanations'].append(f"Gradient norm {gn:.2e} much higher than median ({median_grad:.2e}); likely exploding gradients.")
            elif gn < 1e-5:
                chunk['tags'].append("vanishing_gradient")
                chunk['explanations'].append("Gradient norm very small (<1e-5); likely vanishing gradients.")
        if i > 0 and chunk['train_loss'] is not None and chunks[i-1]['train_loss'] is not None:
            if chunk['train_loss'] > 2 * chunks[i-1]['train_loss']:
                chunk['tags'].append("loss_jump")
                chunk['explanations'].append("Train loss > 2x previous epoch: possible instability.")
        if i > 0 and chunk['train_loss'] is not None and chunks[i-1]['train_loss'] is not None:
            if abs(chunk['train_loss'] - chunks[i-1]['train_loss']) < 0.01:
                chunk['tags'].append("loss_plateau")
                chunk['explanations'].append("Train loss nearly unchanged: learning may be stalled.")
        if i > 0 and chunk['lr'] is not None and chunks[i-1]['lr'] is not None:
            if abs(chunk['lr'] - chunks[i-1]['lr']) > 1e-8:
                chunk['tags'].append("lr_change")
                chunk['explanations'].append(f"Learning rate changed from {chunks[i-1]['lr']} to {chunk['lr']}.")
        if chunk['acc'] is not None:
            random_baseline = 1.0 / num_classes
            if chunk['acc'] < random_baseline + 0.1:
                chunk['tags'].append("acc_stuck")
                chunk['explanations'].append("Accuracy stuck near random chance; may indicate underfitting or data issue.")

def analyze_global(chunks):
    n = len(chunks)
    train_losses = [c['train_loss'] for c in chunks if c['train_loss'] is not None]
    val_losses = [c['val_loss'] for c in chunks if c['val_loss'] is not None]
    accs = [c['acc'] for c in chunks if c['acc'] is not None]
    tags_flat = sum([c['tags'] for c in chunks], [])

    unstable = any(t in tags_flat for t in ("instability", "loss_jump", "exploding_gradient"))
    all_acc_stuck = all('acc_stuck' in c['tags'] for c in chunks if c.get('acc') is not None)
    underfit = train_losses and val_losses and train_losses[-1] > 0.8 * train_losses[0] and val_losses[-1] > 0.8 * val_losses[0]
    overfit = train_losses and val_losses and train_losses[-1] < train_losses[0] - 0.5 and val_losses[-1] > val_losses[0] + 0.2

    analysis = {"unstable": unstable, "all_acc_stuck": all_acc_stuck, "underfit": underfit, "overfit": overfit}
    if unstable:
        print("\n[Global] Training instability or exploding gradients detected. Address stability first before other changes.")
    if all_acc_stuck:
        print("\n[Global] Accuracy stuck near random chance for all epochs. Data or label problem likely.")
    if underfit and not unstable and not all_acc_stuck:
        print("\n[Global] Underfitting detected: both train and val loss are high or flat.")
        print("Suggestion: Try larger model, higher LR, train longer.\n")
    if overfit:
        print("\n[Global] Overfitting detected: train loss dropping but val loss rising.")
        print("Suggestion: Increase dropout/weight decay or use early stopping.\n")
    return analysis

def print_chunk_results(chunks):
    print("\n--- Per-epoch (chunk) analysis ---")
    for chunk in chunks:
        summary = f"Epoch {chunk['epoch']} | train_loss={chunk['train_loss']} val_loss={chunk['val_loss']} acc={chunk['acc']} lr={chunk['lr']} grad_norm={chunk.get('grad_norm', None)}"
        if chunk['tags']:
            print(summary)
            for tag, expl in zip(chunk['tags'], chunk['explanations']):
                print(f"  [{tag}] {expl}")
        else:
            print(summary + " | OK")
    print("--- End of epoch analysis ---\n")

def process_from_file(path, num_classes=2):
    with open(path, "r") as f:
        lines = f.readlines()
    chunks, grad_norms = parse_log_into_chunks(lines)
    analyze_chunks(chunks, grad_norms, num_classes=num_classes)
    print_chunk_results(chunks)
    return analyze_global(chunks)

def process_from_stdin(num_classes=2):
    print("Paste your training log output and hit Ctrl-D (EOF):")
    lines = sys.stdin.read().splitlines()
    chunks, grad_norms = parse_log_into_chunks(lines)
    analyze_chunks(chunks, grad_norms, num_classes=num_classes)
    print_chunk_results(chunks)
    return analyze_global(chunks)

# Optionally, you can create a LogReport object for downstream use (not shown here)
