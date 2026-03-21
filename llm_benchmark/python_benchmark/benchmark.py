import time
import statistics
import platform

import torch

from model import Gpt, GptConfig


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


def benchmark_model(
    model: Gpt,
    config: GptConfig,
    batch_size: int,
    sequence_length: int,
    num_iterations: int,
    device: torch.device,
    backend_name: str,
    model_name: str,
):
    model.eval()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
        device=device,
        dtype=torch.long,
    )

    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)
    sync_device(device)

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        sync_device(device)
        end = time.perf_counter()
        times.append(end - start)

    total_time = sum(times)
    avg_time = total_time / num_iterations
    throughput = (batch_size * sequence_length * num_iterations) / total_time
    p50_us = statistics.median(times) * 1e6

    print()
    print("╔" + "═" * 59 + "╗")
    print(f"║ Benchmark Results: {backend_name:<39}║")
    print("╠" + "═" * 59 + "╣")
    print(f"║ Model Size: {model_name:>48} ║")
    print(f"║ Batch Size: {batch_size:>47} ║")
    print(f"║ Sequence Length: {sequence_length:>42} ║")
    print(f"║ Iterations: {num_iterations:>47} ║")
    print("╟" + "─" * 59 + "╢")
    print(f"║ Total Time: {total_time * 1e6:>41.2f}µs ║")
    print(f"║ Avg Time/Iter: {avg_time * 1e6:>37.2f}µs ║")
    print(f"║ p50 latency: {p50_us:>40.2f} µs ║")
    print(f"║ Throughput: {throughput:>33.2f} tok/s ║")
    print("╚" + "═" * 59 + "╝")

    return {
        "backend": backend_name,
        "model": model_name,
        "avg_time_s": avg_time,
        "p50_us": p50_us,
        "throughput_tok_s": throughput,
    }


def compare_results(results):
    print()
    print("╔" + "═" * 79 + "╗")
    print("║ PYTORCH BENCHMARK COMPARISON".ljust(80) + "║")
    print("╠" + "═" * 79 + "╣")
    print("║ Backend              │ Model      │ Avg Time     │ p50 µs     │ Throughput   ║")
    print("╟" + "─" * 79 + "╢")
    for r in results:
        avg_us = r["avg_time_s"] * 1e6
        print(
            f"║ {r['backend']:<20} │ "
            f"{r['model']:<10} │ "
            f"{avg_us:>10.2f}µs │ "
            f"{r['p50_us']:>10.0f} │ "
            f"{r['throughput_tok_s']:>11.2f} tok/s ║"
        )
    print("╚" + "═" * 79 + "╝")


def main():
    print("PyTorch Benchmarking Suite\n")
    print(f"OS: {platform.system().lower()}")
    print(f"Architecture: {platform.machine()}")

    device = pick_device()
    print(f"PyTorch device: {device}\n")

    model_configs = [
        ("tiny", GptConfig.tiny()),
        ("small", GptConfig.small()),
    ]

    batch_size = 4
    sequence_length = 32
    num_iterations = 50

    results = []

    backend_name = f"PyTorch ({device.type.upper()})"

    for model_name, cfg in model_configs:
        print("-" * 60)
        print(
            f"Model: {model_name.upper()} "
            f"(vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, "
            f"layers={cfg.num_layers}, heads={cfg.num_heads})"
        )
        print("-" * 60)

        model = Gpt(cfg).to(device)
        result = benchmark_model(
            model=model,
            config=cfg,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_iterations=num_iterations,
            device=device,
            backend_name=backend_name,
            model_name=model_name,
        )
        results.append(result)

    compare_results(results)


if __name__ == "__main__":
    main()