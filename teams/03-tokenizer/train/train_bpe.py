#!/usr/bin/env python3
"""
train_bpe.py
============

Builds a 131,072-id byte-level BPE tokenizer with a fixed id slot per Brahmic
script, in the BrahmicTokenizer-131K format (HF tokenizer.json, o200k_base
pre-tokenizer, GPT-2 ByteLevel).

Id layout (from slots.json)
---------------------------
    [0, base)                    o200k_base tokens in rank order (English, code, ...)
    [base, base + slots_1)       new merges for script 1 (e.g. Devanagari)
    ...                          one contiguous block per script, in slots.json order
    [vocab - special_slots, vocab)  special tokens: named ones, then <|reserved_N|>

Method
------
1. Base. Walk o200k_base in rank order. Keep a token only if it is <= 32 bytes,
   spans at most one script, and both halves of its merge were kept. Stop at
   `base` tokens. The kept set is closed under its merges, so it is a valid BPE.
2. Per script. Pre-tokenize the script's corpus with the o200k regex, keep only
   pre-tokens whose non-Common characters are all in that script, encode them
   with the base, and continue BPE training on those words for exactly `slots`
   merges. A merge is allowed only if the result is <= 32 bytes, spans one
   script, and is not already in the vocab.
3. Specials fill the top `special_slots` ids.

Steps 1 and 2 enforce the four verify-script properties by construction:
no cross-script merges, no normal token > 32 bytes. With --brahmic-dir the
script runs the four verify scripts on the output as a final gate. The
structural-identity script compares the output with --compare-to (default:
the BrahmicTokenizer-131K tokenizer.json), not with itself; it reports the
vocab / merge / special deltas and does not fail on a difference.

Usage
-----
    python train_bpe.py --corpus /path/to/corpus --out /path/to/out \\
        --brahmic-dir /path/to/BrahmicTokenizer-131K

Writes <out>/tokenizer.json and <out>/manifest.json (sha256, slot table,
corpus hashes, verify results). Exit 1 if a verify script fails.
"""
import argparse
import hashlib
import heapq
import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

MAX_BYTES = 32

# Script ranges and name prefixes: the union of the classifiers in
# verify_no_cross_script_merges.py and verify_kronecker_constraints_unified.py,
# so a token we accept is accepted by both.
INDIC_RANGES = {
    "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF), "Oriya": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F),
    "Sinhala": (0x0D80, 0x0DFF), "Arabic": (0x0600, 0x06FF),
}
NAME_PREFIXES = [("LATIN", "Latin"), ("CJK", "CJK"), ("HIRAGANA", "CJK"), ("KATAKANA", "CJK"),
                 ("HANGUL", "CJK"), ("CYRILLIC", "Cyrillic"), ("THAI", "Thai"), ("ARABIC", "Arabic"),
                 ("SINHALA", "Sinhala"), ("HEBREW", "Hebrew"), ("GREEK", "Greek")]


def script_of_char(ch):
    cp = ord(ch)
    for s, (lo, hi) in INDIC_RANGES.items():
        if lo <= cp <= hi:
            return s
    name = unicodedata.name(ch, "")
    for prefix, s in NAME_PREFIXES:
        if name.startswith(prefix):
            return s
    return None


def scripts_of(b):
    """Non-Common scripts in a byte string, decoded the way the verify scripts decode it."""
    return {s for ch in b.decode("utf-8", errors="replace") if (s := script_of_char(ch)) is not None}


def gpt2_byte_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = gpt2_byte_map()
U2B = {u: b for b, u in B2U.items()}


def to_unicode(b):
    return "".join(B2U[x] for x in b)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def o200k_split(ranks, token, rank):
    """The two parts o200k_base merges into `token`: BPE on its bytes using ranks < rank."""
    parts = [bytes([x]) for x in token]
    while len(parts) > 2:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < rank and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
    return parts


def build_base(size):
    """Step 1: the first `size` admissible o200k_base tokens and their merges."""
    import tiktoken
    ranks = tiktoken.get_encoding("o200k_base")._mergeable_ranks
    vocab, merges = {}, []          # bytes -> id, [(left bytes, right bytes)]
    dropped = Counter()
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(vocab) == size:
            break
        if len(token) == 1:
            vocab[token] = len(vocab)
            continue
        if len(token) > MAX_BYTES:
            dropped["over_32_bytes"] += 1
            continue
        if len(scripts_of(token)) > 1:
            dropped["cross_script"] += 1
            continue
        parts = o200k_split(ranks, token, rank)
        if len(parts) != 2 or parts[0] not in vocab or parts[1] not in vocab:
            dropped["parent_dropped"] += 1
            continue
        vocab[token] = len(vocab)
        merges.append((parts[0], parts[1]))
    if len(vocab) != size or sum(len(t) == 1 for t in vocab) != 256:
        sys.exit(f"ERROR: base has {len(vocab)} tokens, wanted {size} including all 256 bytes.")
    return vocab, merges, dict(dropped)


def tokenizer_json(vocab, merges, specials, pattern):
    """vocab: unicode-string -> id; merges: [(a, b)] unicode strings; specials: [(id, content)]."""
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for i, c in specials],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": [[a, b] for a, b in merges]},
    }


def word_counts(paths, script, pattern):
    """Pre-token byte strings from the corpus whose non-Common characters are all in `script`."""
    from tokenizers import Regex, pre_tokenizers
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    counts = Counter()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                counts.update(piece for piece, _ in split.pre_tokenize_str(line))
    keep = Counter()
    for piece, c in counts.items():
        if {script_of_char(ch) for ch in piece} - {None} == {script}:
            keep[piece.encode("utf-8")] = c
    return keep


def learn_merges(words, freqs, id_bytes, vocab_bytes, script, n_merges, first_id):
    """Step 2: continue BPE on `words` (lists of ids) for exactly n_merges new tokens."""
    pair_count = defaultdict(int)
    pair_words = defaultdict(set)
    for wi, w in enumerate(words):
        for a, b in zip(w, w[1:]):
            pair_count[(a, b)] += freqs[wi]
            pair_words[(a, b)].add(wi)
    heap = [(-c, a, b) for (a, b), c in pair_count.items()]
    heapq.heapify(heap)
    banned = set()
    merges = []
    next_id = first_id
    while len(merges) < n_merges:
        if not heap:
            sys.exit(f"ERROR: {script} ran out of pairs after {len(merges)} merges; "
                     f"lower its slots or add corpus.")
        negc, a, b = heapq.heappop(heap)
        if -negc != pair_count.get((a, b), 0) or (a, b) in banned or -negc <= 0:
            continue
        new = id_bytes[a] + id_bytes[b]
        if len(new) > MAX_BYTES or new in vocab_bytes or scripts_of(new) - {script}:
            banned.add((a, b))
            continue
        nid = next_id
        next_id += 1
        id_bytes[nid] = new
        vocab_bytes.add(new)
        merges.append((a, b, nid))
        touched = set()
        for wi in pair_words.pop((a, b)):
            w = words[wi]
            if len(w) < 2:
                continue
            out, i, hit = [], 0, False
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(nid)
                    i += 2
                    hit = True
                else:
                    out.append(w[i])
                    i += 1
            if not hit:
                continue
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_count[p] -= f
                touched.add(p)
            for p in zip(out, out[1:]):
                pair_count[p] += f
                pair_words[p].add(wi)
                touched.add(p)
            words[wi] = out
        pair_count.pop((a, b), None)
        for p in touched:
            c = pair_count.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p[0], p[1]))
    return merges


def run_verify(brahmic_dir, tok_path, compare_to):
    """Run the four reviewer scripts; the structural diff is against `compare_to`, not the output itself."""
    cmds = [
        ["verify_no_cross_script_merges.py", tok_path],
        ["verify_max_byte_length.py", tok_path],
        ["verify_kronecker_constraints_unified.py", tok_path],
        ["verify_sarvam_m_structural_identity.py", tok_path, compare_to],
    ]
    results = {}
    for cmd in cmds:
        p = subprocess.run([sys.executable, str(Path(brahmic_dir) / cmd[0]), *map(str, cmd[1:])],
                           capture_output=True, text=True)
        out = p.stdout + p.stderr
        # verify_kronecker exits 0 even on a failed constraint, so read its RESULT line too.
        ok = p.returncode == 0 and "RESULT: FAIL" not in out and "FAIL:" not in out
        results[cmd[0]] = {"exit": p.returncode, "pass": ok,
                           "summary": [ln.strip() for ln in out.splitlines()
                                       if any(k in ln for k in ("PASS", "FAIL", "RESULT", " identical:"))]}
        label = "REPORTED" if ok and cmd[0].startswith("verify_sarvam") else ("PASS" if ok else "FAIL")
        print(f"  {cmd[0]:<42} exit={p.returncode}  {label}", flush=True)
        if not ok:
            print(out[-2000:], flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help="directory with <lang>.txt from fetch_corpus.py")
    ap.add_argument("--out", required=True, help="output directory (outside the repo; tokenizer.json > 5 MB)")
    ap.add_argument("--slots", default=str(Path(__file__).with_name("slots.json")))
    ap.add_argument("--brahmic-dir", help="clone of BrahmicTokenizer-131K; runs its four verify scripts on the output")
    ap.add_argument("--compare-to", help="tokenizer.json for the structural diff (default: <brahmic-dir>/tokenizer.json)")
    args = ap.parse_args()

    import tiktoken
    t_start = time.time()
    cfg = json.loads(Path(args.slots).read_text())
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pattern = tiktoken.get_encoding("o200k_base")._pat_str

    vocab_size, n_special = cfg["vocab_size"], cfg["special_slots"]
    scripts = cfg["scripts"]
    base_size = vocab_size - n_special - sum(s["slots"] for s in scripts.values())
    print(f"layout: base={base_size}  scripts={sum(s['slots'] for s in scripts.values())}  "
          f"specials={n_special}  total={vocab_size}", flush=True)

    t0 = time.time()
    base_vocab, base_merges, dropped = build_base(base_size)
    print(f"base: {len(base_vocab)} o200k tokens, {len(base_merges)} merges, dropped {dropped} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # Encoder for the base alone, used to segment each script's words before step 2.
    from tokenizers import Tokenizer
    base_json = tokenizer_json({to_unicode(b): i for b, i in base_vocab.items()},
                               [(to_unicode(a), to_unicode(b)) for a, b in base_merges], [], pattern)
    base_json["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}
    base_tok = Tokenizer.from_str(json.dumps(base_json))

    id_bytes = {i: b for b, i in base_vocab.items()}
    vocab_bytes = set(base_vocab)
    all_merges = [(base_vocab[a], base_vocab[b]) for a, b in base_merges]
    layout = {"base": [0, base_size]}
    corpus_manifest_path = corpus / "corpus_manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text()) if corpus_manifest_path.exists() else {}
    next_id = base_size
    for script, spec in scripts.items():
        t0 = time.time()
        paths = [corpus / f"{lang}.txt" for lang in spec["langs"]]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            sys.exit(f"ERROR: missing corpus files {missing}. Run fetch_corpus.py first.")
        counts = word_counts(paths, script, pattern)
        items = sorted(counts.items())                      # fixed order -> deterministic ties
        encs = base_tok.encode_batch([b.decode("utf-8") for b, _ in items], add_special_tokens=False)
        words = [e.ids for e in encs]
        freqs = [c for _, c in items]
        merges = learn_merges(words, freqs, id_bytes, vocab_bytes, script, spec["slots"], next_id)
        all_merges.extend((a, b) for a, b, _ in merges)
        layout[script] = [next_id, next_id + spec["slots"]]
        print(f"{script}: {len(items)} unique words, {sum(freqs)} occurrences -> ids "
              f"[{next_id}, {next_id + spec['slots']})  ({time.time() - t0:.0f}s)", flush=True)
        next_id += spec["slots"]

    names = cfg["named_specials"] + [f"<|reserved_{i}|>" for i in range(n_special - len(cfg["named_specials"]))]
    specials = [(next_id + i, c) for i, c in enumerate(names)]
    layout["specials"] = [next_id, next_id + n_special]
    assert next_id + n_special == vocab_size

    vocab = {to_unicode(b): i for i, b in id_bytes.items()}
    vocab.update({c: i for i, c in specials})
    merges_u = [(to_unicode(id_bytes[a]), to_unicode(id_bytes[b])) for a, b in all_merges]
    tok_path = out / "tokenizer.json"
    tok_path.write_text(json.dumps(tokenizer_json(vocab, merges_u, specials, pattern), ensure_ascii=False), encoding="utf-8")

    # Load back through the real library and round-trip a mixed sample.
    tok = Tokenizer.from_file(str(tok_path))
    sample = ["भारत एक देश है", "আমার সোনার বাংলা", "def f(x):\n    return x ** 2  # ok", "ଓଡ଼ିଆ ଭାଷା", "Hello, world!"]
    for lang in ("Hi", "Ta", "Or"):
        with open(corpus / f"{lang}.txt", encoding="utf-8") as f:
            sample.append(f.readline())
    for s in sample:
        if tok.decode(tok.encode(s).ids) != s:
            sys.exit(f"ERROR: round-trip failed on {s[:60]!r}")
    assert tok.get_vocab_size() == vocab_size

    sha = sha256_file(tok_path)
    manifest = {
        "tokenizer": "tokenizer.json",
        "sha256": sha,
        "vocab_size": vocab_size,
        "merges": len(merges_u),
        "layout": layout,
        "slots": {s: v["slots"] for s, v in scripts.items()},
        "special_tokens": names[:len(cfg["named_specials"])],
        "reserved_special_slots": n_special - len(cfg["named_specials"]),
        "base": {"source": "tiktoken o200k_base", "kept": base_size, "dropped": dropped},
        "corpus": {lang: corpus_manifest.get(lang) for spec in scripts.values() for lang in spec["langs"]},
        "slots_json_sha256": sha256_file(args.slots),
        "build_seconds": round(time.time() - t_start),
    }
    print(f"wrote {tok_path}  sha256={sha}", flush=True)

    ok = True
    if args.brahmic_dir:
        print("verify scripts:", flush=True)
        compare_to = args.compare_to or str(Path(args.brahmic_dir) / "tokenizer.json")
        manifest["verify"] = run_verify(args.brahmic_dir, tok_path, compare_to)
        manifest["structural_diff_against"] = args.compare_to or "BrahmicTokenizer-131K/tokenizer.json"
        ok = all(v["pass"] for v in manifest["verify"].values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {out / 'manifest.json'}  ({manifest['build_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
