"""Build the paper's FineWeb-Edu token files from local Parquet shards."""

from contextlib import ExitStack
import json
from pathlib import Path
import pickle
import struct
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


# Fill in these paths before running. Relative paths are resolved from this file.
PARQUET_DIR = ""
TOKENIZER_DIR = ""
TRAIN_BIN_PATH = ""
VAL_BIN_PATH = ""
META_PKL_PATH = ""

# Keep the shard order and shuffle settings fixed to reproduce the paper data.
PARQUET_NAMES = tuple(
    f"{part:03d}_{shard:05d}.parquet"
    for part in range(2)
    for shard in range(10)
)
VOCAB_SIZE = 49_152
EOS_TOKEN_ID = 0
VAL_FRACTION_BY_TOKENS = 0.001
SHUFFLE_SEED = 2605
SHUFFLE_BUCKET_BITS = 8
BIN_DTYPE = np.dtype("<u2")
RECORD_HEADER = struct.Struct("<QQI")  # shuffle key, document ordinal, token count
WRITE_BUFFER_BYTES = 8_000_000
SCRIPT_DIR = Path(__file__).resolve().parent


def configured_path(value, setting):
    if not value.strip():
        raise ValueError(f"Set {setting} at the top of this script.")
    return (SCRIPT_DIR / Path(value).expanduser()).resolve()


def check_parquet_files(paths):
    for path in tqdm(paths, desc="Checking Parquet shards"):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty Parquet shard: {path}")
        with pq.ParquetFile(path) as shard:
            if "text" not in shard.schema.names:
                raise ValueError(f"No text column in {path}")
            group_count = shard.metadata.num_row_groups
            if group_count == 0 or shard.metadata.num_rows == 0:
                raise ValueError(f"No rows in {path}")
            for group in sorted({0, group_count // 2, group_count - 1}):
                batches = shard.iter_batches(batch_size=128, row_groups=[group], columns=["text"])
                sample = next(batches, None)
                if sample is not None and any(sample.column(0).to_pylist()):
                    break
            else:
                raise ValueError(f"No nonempty text in the sampled rows of {path}")


def document_key(file_index, row_group_index, row_index):
    # Use the same positional key and SplitMix64 mixing as the original build.
    mask = (1 << 64) - 1
    x = SHUFFLE_SEED & mask
    x ^= (file_index + 1) * 0xD6E8FEB86659FD93
    x ^= (row_group_index + 1) * 0xA5A3564E27F886B7
    x ^= (row_index + 1) * 0x8CB92BA72F3D8DD7
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


def write_shuffle_buckets(parquet_paths, tokenizer, bucket_paths):
    documents = 0
    tokens = 0
    with ExitStack() as stack:
        buckets = [stack.enter_context(path.open("wb")) for path in bucket_paths]
        for file_index, path in enumerate(tqdm(parquet_paths, desc="Tokenizing shards")):
            with pq.ParquetFile(path) as shard:
                for group_index in range(shard.metadata.num_row_groups):
                    table = shard.read_row_group(group_index, columns=["text"])
                    for row_index, text in enumerate(table.column("text").to_pylist()):
                        if not text:
                            continue
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        ids.append(tokenizer.eos_token_id)
                        key = document_key(file_index, group_index, row_index)
                        bucket = buckets[key >> (64 - SHUFFLE_BUCKET_BITS)]
                        bucket.write(RECORD_HEADER.pack(key, documents, len(ids)))
                        bucket.write(np.asarray(ids, dtype=BIN_DTYPE).tobytes())
                        documents += 1
                        tokens += len(ids)
    return documents, tokens


def read_bucket(path):
    records = []
    with path.open("rb") as bucket:
        while header := bucket.read(RECORD_HEADER.size):
            if len(header) != RECORD_HEADER.size:
                raise ValueError(f"Incomplete record header in {path}")
            key, ordinal, count = RECORD_HEADER.unpack(header)
            payload_size = count * BIN_DTYPE.itemsize
            payload = bucket.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"Incomplete token payload in {path}")
            records.append((key, ordinal, count, payload))
    records.sort(key=lambda record: (record[0], record[1]))
    return records


def shuffled_documents(bucket_paths):
    # High key bits select the bucket, so this yields a globally sorted stream.
    for path in tqdm(bucket_paths, desc="Writing validation and training tokens"):
        for _, _, count, payload in read_bucket(path):
            yield count, payload


def write_split(path, documents, created, token_target=None):
    output = path.open("xb", buffering=WRITE_BUFFER_BYTES)
    created.append(path)
    doc_count = token_count = 0
    with output:
        for count, payload in documents:
            output.write(payload)
            doc_count += 1
            token_count += count
            if token_target is not None and token_count >= token_target:
                break
    return doc_count, token_count


def main():
    parquet_dir = configured_path(PARQUET_DIR, "PARQUET_DIR")
    tokenizer_dir = configured_path(TOKENIZER_DIR, "TOKENIZER_DIR")
    train_path = configured_path(TRAIN_BIN_PATH, "TRAIN_BIN_PATH")
    val_path = configured_path(VAL_BIN_PATH, "VAL_BIN_PATH")
    meta_path = configured_path(META_PKL_PATH, "META_PKL_PATH")
    outputs = (train_path, val_path, meta_path)
    if len(set(outputs)) != len(outputs) or any(
        a in b.parents for a in outputs for b in outputs
    ):
        raise ValueError("The three output paths must be distinct files.")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    parquet_paths = [parquet_dir / name for name in PARQUET_NAMES]
    check_parquet_files(parquet_paths)
    model_config = json.loads((tokenizer_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    if (
        model_config["vocab_size"] != VOCAB_SIZE
        or len(tokenizer) != VOCAB_SIZE
        or tokenizer.eos_token_id != EOS_TOKEN_ID
        or not tokenizer.is_fast
    ):
        raise ValueError("Expected the paper's fast tokenizer (49152 tokens, EOS ID 0).")

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    created = []
    try:
        with TemporaryDirectory(prefix=".fineweb-shuffle-", dir=train_path.parent) as work_dir:
            bucket_count = 1 << SHUFFLE_BUCKET_BITS
            bucket_paths = [Path(work_dir) / f"bucket_{i:04d}.bin" for i in range(bucket_count)]
            total_docs, total_tokens = write_shuffle_buckets(parquet_paths, tokenizer, bucket_paths)
            if total_docs < 2:
                raise ValueError("Not enough documents for a train/validation split.")

            target_val_tokens = max(1, round(total_tokens * VAL_FRACTION_BY_TOKENS))
            documents = shuffled_documents(bucket_paths)
            try:
                val_docs, val_tokens = write_split(val_path, documents, created, target_val_tokens)
                train_docs, train_tokens = write_split(train_path, documents, created)
            finally:
                documents.close()
            if train_docs == 0 or val_docs == 0:
                raise ValueError("The split produced an empty training or validation set.")
            if train_docs + val_docs != total_docs or train_tokens + val_tokens != total_tokens:
                raise RuntimeError("Output counts do not match the shuffled document stream.")

        meta = {
            "dataset": "FineWeb-Edu sample/100BT",
            "source_files": list(PARQUET_NAMES),
            "vocab_size": VOCAB_SIZE,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_size": len(tokenizer),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "dtype": BIN_DTYPE.str,
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_buckets": bucket_count,
            "val_fraction_by_tokens_target": VAL_FRACTION_BY_TOKENS,
            "target_val_tokens": target_val_tokens,
            "train_docs": train_docs,
            "train_tokens": train_tokens,
            "val_docs": val_docs,
            "val_tokens": val_tokens,
            "total_docs": total_docs,
            "total_tokens": total_tokens,
        }
        with meta_path.open("xb") as output:
            created.append(meta_path)
            pickle.dump(meta, output)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    print(f"Training:   {train_tokens:,} tokens from {train_docs:,} documents")
    print(f"Validation: {val_tokens:,} tokens from {val_docs:,} documents")


if __name__ == "__main__":
    main()
