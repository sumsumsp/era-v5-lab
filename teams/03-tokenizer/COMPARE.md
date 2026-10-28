# COMPARE: tokenizer comparison (Week 1)

Produced by `harness/run_harness.py --tokenizers all --suites flores,code --check-reference` on 27 Oct 2026.
HF tokenizers are pinned to the commit in `REGISTRY` (`harness/run_harness.py`), all load with `trust_remote_code=False` and no HF login.

Tokenizer v0 (`v0/`, #107) only renames reserved special slots, so it encodes plain text to the same ids and has the same numbers as the BrahmicTokenizer-131K rows.

**Pass bar:** beat BrahmicTokenizer-131K on mean Indic fertility; at most 2% worse on English (FLORES En fertility), HumanEval and MBPP (tokens / char).

## FLORES-200 fertility (tokens / word, lower is better; Mean = 11 Indic languages)

| Tokenizer | HF repo | En | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BrahmicTokenizer-131K** | theschoolofai/BrahmicTokenizer-131K | 1.2352 | 1.6429 | 2.3295 | 3.1166 | 3.0292 | 3.2734 | 3.4738 | 2.5723 | 2.2748 | 2.5955 | 4.1287 | 2.7535 | **2.8355** |
| o200k_base | tiktoken | 1.2229 | 1.6432 | 2.3330 | 3.1496 | 3.0486 | 3.3039 | 3.4955 | 2.5741 | 2.2901 | 2.7187 | 6.7948 | 2.7624 | 3.1013 |
| Gemma-4 | google/gemma-4-E4B | 1.2366 | 1.3857 | 1.6764 | 2.3649 | 2.8299 | 3.2606 | 3.3266 | 1.9986 | 2.4006 | 2.8237 | 4.8339 | 2.8158 | 2.7015 |
| Qwen3.8 | Qwen/Qwen3.8-27B | 1.2530 | 2.1566 | 3.0695 | 4.2182 | 4.6199 | 5.3352 | 5.5795 | 3.3347 | 4.0362 | 4.0662 | 5.4188 | 4.2060 | 4.1855 |
| Qwen3-8B | Qwen/Qwen3-8B | 1.2572 | 4.7500 | 7.0953 | 9.9963 | 11.3522 | 11.8438 | 13.2906 | 6.6980 | 8.8635 | 7.7576 | 13.5997 | 7.7003 | 9.3588 |
| DeepSeek-V4 | deepseek-ai/DeepSeek-V4-Flash-Base | 1.2306 | 2.9477 | 2.7854 | 4.9161 | 5.9688 | 6.2065 | 7.9431 | 4.2720 | 4.9805 | 4.6761 | 7.6055 | 3.7633 | 5.0968 |
| Sarvam-30B | sarvamai/sarvam-30b | 1.2368 | 1.3857 | 1.6765 | 2.3649 | 2.3186 | 2.5618 | 2.9350 | 1.9986 | 1.9340 | 1.6313 | 1.9659 | 2.8159 | 2.1444 |
| Sarvam-m | sarvamai/sarvam-m | 1.2672 | 1.9405 | 2.9155 | 3.5911 | 3.7030 | 3.8345 | 4.7443 | 3.1445 | 3.6214 | 3.1976 | 18.1782 | 4.7235 | 4.8722 |
| Sarvam-1 | sarvamai/sarvam-1 | 1.4313 | 1.4039 | 2.0467 | 2.1554 | 2.1321 | 2.4001 | 2.8345 | 1.7765 | 1.8024 | 1.6826 | 2.3251 | 4.7451 | 2.3004 |
| c0 (`train/train_bpe.py`) | not published | 1.2596 | 1.6310 | 2.2673 | 2.6597 | 2.5954 | 2.7836 | 3.0869 | 2.1067 | 2.1193 | 2.0053 | 2.6162 | 2.3402 | 2.3829 |

## Code and math (tokens / char, lower is better)

| Tokenizer | HumanEval | MBPP | GSM8K |
|---|---|---|---|
| **BrahmicTokenizer-131K** | 0.2951 | 0.3195 | 0.3010 |
| o200k_base | 0.2934 | 0.3169 | 0.2995 |
| Gemma-4 | 0.3340 | 0.3876 | 0.3505 |
| Qwen3.8 | 0.3170 | 0.3318 | 0.3525 |
| Qwen3-8B | 0.2980 | 0.3169 | 0.3469 |
| DeepSeek-V4 | 0.2976 | 0.3265 | 0.3015 |
| Sarvam-30B | 0.3340 | 0.3876 | 0.3505 |
| Sarvam-m | 0.3073 | 0.3378 | 0.3508 |
| Sarvam-1 | 0.3780 | 0.4240 | 0.3805 |
| c0 (`train/train_bpe.py`) | 0.2979 | 0.3226 | 0.3038 |

## Cross-checks against the paper's reference outputs

Rows that also appear in `BrahmicTokenizer-131K/reference_outputs/evaluate_fertility_flores.json` match to 4 decimals:
Brahmic, Qwen3-8B, Sarvam-30B, Sarvam-1. Also identical to 4 decimals:

- DeepSeek-V4 = the reference DeepSeek-R1 row: V4 kept the R1 tokenizer.
- Gemma-4 = the reference Gemma-3-1B row: same tokenizer.
- o200k_base Indic columns = the reference GPT-OSS-120B row (GPT-OSS uses o200k).

Sarvam-30B and Gemma-4 tokenize Hindi, Bengali and code identically but differ on Te/Kn/Ml/Gu/Pa/Or, where Sarvam-30B is much better.

## Reading the table

- Two comparators beat BrahmicTokenizer-131K on Indic already: Sarvam-30B (2.14) and Sarvam-1 (2.30). Both are **> 13% worse on code**, so neither passes the bar.
- **c0 passes the bar, with almost no English margin:**
  - Indic: −15.96%
  - HumanEval: +0.95%
  - MBPP: +0.97%
  - English: **+1.98% (limit 2.00%)**

  The English loss comes from keeping only the first 114,432 o200k ranks. Pruning the base by English and code token frequency instead of by rank is the main Week 2 change.
- c0 is the Week 1 pipeline proof, not a release candidate. It is trained on 20 MB of FineWeb-2 per language, and its slot sizes in `train/slots.json` are not tuned.

## Reproduce c0 (about 2 minutes after the download)

```bash
python train/fetch_corpus.py --out $WORK/corpus --mb 20        # FineWeb-2, never FLORES
python train/train_bpe.py --corpus $WORK/corpus --out $WORK/c0 --brahmic-dir $BRAHMIC
#   3 verify scripts PASS + structural diff vs BrahmicTokenizer-131K reported; sha256 a635bc3d601925a80f7792ba73f7aba6867a688a61bb9031481eeed5578173c1
python harness/run_harness.py --tokenizers BrahmicTokenizer-131K --candidate c0=$WORK/c0/tokenizer.json \
    --suites flores,code --check-reference --brahmic-dir $BRAHMIC --flores-dir $WORK/flores200_dataset
```

The build is deterministic: two runs gave the same sha256. Layout, corpus hashes and verify results are in `train/c0_manifest.json`.
The tokenizer.json (5,674,537 bytes) is not committed because pr-guard rejects files over 5 MiB (5,242,880 bytes).
