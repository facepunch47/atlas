#!/usr/bin/env python3
"""Qwen3.8-27B K-sweep: tok/s, TTFT, mean accepted at K=1..4.

Copy of bench/mtp_gate_throughput.py, retargeted. Sequential (batch=1)
thinking-off short decode, same prompt every cell. Engine convention:
K = num_drafts + 1.

This script measures a serve that is ALREADY UP (same as
mtp_gate_throughput.py). Restart between legs — the bench cannot change
live --num-drafts:

  K=1  no --speculative
  K=2  --speculative --num-drafts 1   # unpinned 3.8 / engine default
  K=3  --speculative --num-drafts 2
  K=4  --speculative --num-drafts 3   # atlas-recipes#19 pin

Accept stats: /metrics `atlas_spec_decode_verify_total`, same quantity as
mtp_accept_debug.rs (`mean_na`, `tok_step = 1 + mean_na`). Enable
ATLAS_MTP_ACCEPT_DEBUG on the serve to also get the log flush.

Does NOT pin default_num_drafts. Not a gate.

Usage (one live K, like mtp_gate_throughput.py):
  python bench/k_sweep.py [host] [port] [runs] [max_tokens] [num_drafts]

  num_drafts is a LABEL (0/1/2/3). It does not change the live width.

Usage (restart serve per leg when docker is available):
  LEGS="0 1 2 3" MODEL=unsloth/Qwen3.8-27B-NVFP4 python bench/k_sweep.py --restart
"""
import json
import os
import subprocess
import sys
import time

import requests

PROMPT = (
    "Count from 1 upward, one number per line, until told to stop."
)


def k_width(num_drafts):
    """Engine convention: K = num_drafts + 1."""
    return num_drafts + 1


def outcome_drafts(outcome):
    if outcome in ("reject", "accept-0"):
        return 0
    if outcome in ("accept", "accept-1"):
        return 1
    if outcome.startswith("accept-"):
        try:
            return int(outcome.split("-", 1)[1])
        except ValueError:
            return None
    return None


def parse_verify(text):
    """(k_label, accepted_drafts) -> count from Prometheus text."""
    counts = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("atlas_spec_decode_verify_total"):
            continue
        try:
            labels, rest = line[line.index("{") + 1 : line.index("}")], line[
                line.index("}") + 1 :
            ]
        except ValueError:
            continue
        kv = {}
        for part in labels.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
        if "k" not in kv or "outcome" not in kv:
            continue
        acc = outcome_drafts(kv["outcome"])
        if acc is None:
            continue
        n = rest.split()
        if not n:
            continue
        try:
            counts[(kv["k"], acc)] = counts.get((kv["k"], acc), 0) + int(float(n[0]))
        except ValueError:
            continue
    return counts


def mean_na_and_hist(counts, k):
    hist = {}
    for (label, acc), n in counts.items():
        if label != str(k):
            continue
        hist[acc] = hist.get(acc, 0) + n
    total = sum(hist.values())
    if not total:
        return None, hist
    mean_na = sum(acc * n for acc, n in hist.items()) / total
    return mean_na, hist


def observed_k(counts):
    per = {}
    for (label, _), n in counts.items():
        try:
            per[int(label)] = per.get(int(label), 0) + n
        except ValueError:
            continue
    if not per:
        return None
    return max(per, key=per.get)


def fetch_metrics(base):
    try:
        return requests.get(f"{base}/metrics", timeout=10).text
    except requests.RequestException:
        return ""


def one_decode(url, model, max_tokens, timeout=180):
    """One thinking-off stream. Returns (tok/s, ttft_ms, completion_tokens)."""
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": PROMPT}],
    }
    t0 = time.time()
    ttft = None
    tokens = 0
    server_tps = None
    r = requests.post(url, json=body, stream=True, timeout=timeout)
    r.raise_for_status()
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            if ttft is None:
                ttft = (time.time() - t0) * 1000.0
            tokens += 1
        usage = chunk.get("usage") or {}
        if usage.get("completion_tokens"):
            tokens = int(usage["completion_tokens"])
        tps = usage.get("response_token/s") or usage.get("response_tokens_per_second")
        if tps:
            server_tps = float(tps)
    wall = max(time.time() - t0, 1e-6)
    decode = server_tps if server_tps else (tokens / wall if tokens else 0.0)
    return decode, ttft, tokens


def measure(host, port, runs, max_tokens, num_drafts=None):
    base = f"http://{host}:{port}"
    url = f"{base}/v1/chat/completions"
    models = requests.get(f"{base}/v1/models", timeout=10).json()
    model = models["data"][0]["id"]
    asked_k = k_width(num_drafts) if num_drafts is not None else None
    print(
        f"model={model} url={url} runs={runs} max_tokens={max_tokens} "
        f"thinking=off prompt=same"
        + (f" num_drafts={num_drafts} asked_K={asked_k}" if asked_k else ""),
        flush=True,
    )
    before = parse_verify(fetch_metrics(base))
    tpss, ttfts, cts = [], [], []
    for i in range(runs):
        tps, ttft, ct = one_decode(url, model, max_tokens)
        tpss.append(tps)
        if ttft is not None:
            ttfts.append(ttft)
        cts.append(ct)
        ttft_s = f"{ttft:7.1f} ms" if ttft is not None else "      —"
        print(
            f"  run {i+1:2d}: decode={tps:6.2f} tok/s  TTFT={ttft_s}  tokens={ct}",
            flush=True,
        )
    after = parse_verify(fetch_metrics(base))
    delta = {}
    for key, n in after.items():
        d = n - before.get(key, 0)
        if d > 0:
            delta[key] = d
    live_k = observed_k(delta) or asked_k
    mean_na, hist = mean_na_and_hist(delta, live_k) if live_k else (None, {})
    valid = sorted(t for t in tpss if t > 0)
    n = len(valid)
    mean = sum(valid) / n if n else 0.0
    median = valid[n // 2] if n else 0.0
    ttft_med = sorted(ttfts)[len(ttfts) // 2] if ttfts else None
    hist_s = " ".join(f"{a}:{hist[a]}" for a in sorted(hist)) if hist else "—"
    ttft_part = f"{ttft_med:.1f} ms" if ttft_med is not None else "—"
    print(
        f"\nN={n}  mean={mean:.2f} tok/s  median={median:.2f} tok/s  "
        f"TTFT_med={ttft_part}  ",
        end="",
    )
    if mean_na is not None:
        print(
            f"live_K={live_k}  mean_na={mean_na:.3f}  tok_step={1.0 + mean_na:.3f}  "
            f"hist={hist_s}",
            flush=True,
        )
    else:
        print(
            f"live_K={live_k or '—'}  mean_na=—  "
            f"(K=1 is spec-off; else enable /metrics or ATLAS_MTP_ACCEPT_DEBUG)",
            flush=True,
        )
    if asked_k and live_k and asked_k != live_k:
        print(
            f"WARN: asked K={asked_k} but /metrics moved K={live_k}. "
            f"Restart serve (K=1: no --speculative). TUI MTP k= is not SSOT (#488).",
            flush=True,
        )
    print("No default_num_drafts pin.", flush=True)
    return mean, ttft_med, mean_na, live_k


def serve_flags(nd):
    if nd == 0:
        return "--disable-thinking"
    return f"--speculative --num-drafts {nd} --mtp-quantization bf16 --disable-thinking"


def restart_leg(nd, host, port, model, img, hfcache):
    """Docker restart for one num_drafts. K=1 omits --speculative."""
    k = k_width(nd)
    name = f"atlas-ksweep-nd{nd}-k{k}"
    subprocess.call(
        ["docker", "ps", "-q", "--filter", "name=atlas-ksweep-"],
        stdout=subprocess.DEVNULL,
    )
    for c in subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", "name=atlas-ksweep-"],
        text=True,
    ).split():
        subprocess.call(["docker", "rm", "-f", c], stdout=subprocess.DEVNULL)
    time.sleep(2)
    flags = serve_flags(nd).split()
    cmd = [
        "docker", "run", "-d", "--name", name, "--network", "host",
        "--gpus", "all", "--ipc=host",
        "-v", f"{hfcache}:/root/.cache/huggingface",
        "-e", "ATLAS_MTP_ACCEPT_DEBUG=1",
        img, "serve", model, "--host", "0.0.0.0", "--port", str(port),
        "--max-seq-len", "8192", "--max-batch-size", "1",
        "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.70",
        *flags,
    ]
    print(f"===== LEG num_drafts={nd} → K={k} =====\n  {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    for _ in range(150):
        try:
            requests.get(f"http://{host}:{port}/v1/models", timeout=4)
            return
        except requests.RequestException:
            time.sleep(5)
    raise SystemExit(f"serve did not come up for num_drafts={nd}")


def main(argv):
    restart = "--restart" in argv
    argv = [a for a in argv if a != "--restart"]
    host = argv[1] if len(argv) > 1 else "localhost"
    port = argv[2] if len(argv) > 2 else "8888"
    runs = int(argv[3]) if len(argv) > 3 else 2
    max_tokens = int(argv[4]) if len(argv) > 4 else 64
    nd_label = int(argv[5]) if len(argv) > 5 else None

    if restart:
        model = os.environ.get("MODEL", "unsloth/Qwen3.8-27B-NVFP4")
        img = os.environ.get("IMG", "avarok/atlas-gb10:latest")
        hfcache = os.environ.get("HFCACHE", os.path.expanduser("~/.cache/huggingface"))
        legs = [int(x) for x in os.environ.get("LEGS", "0 1 2 3").split()]
        print(
            "### K-sweep  K=num_drafts+1  thinking-off  short decode  same prompt",
            flush=True,
        )
        print("### unpinned 3.8 → K=2; recipe pin num_drafts=3 → K=4", flush=True)
        for nd in legs:
            restart_leg(nd, host, port, model, img, hfcache)
            measure(host, port, runs, max_tokens, nd)
        print("SWEEP_DONE  no default_num_drafts pin", flush=True)
        return

    measure(host, port, runs, max_tokens, nd_label)


if __name__ == "__main__":
    main(sys.argv)
