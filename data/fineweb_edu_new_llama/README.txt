Place train.bin, val.bin, and meta.pkl in this directory.
-----------------------------------------------
To reproduce the dataset used in the paper:

1. Download the 20 required FineWeb-Edu Parquet shards from:
   https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/100BT
   Use only 000_00000.parquet through 000_00009.parquet and
   001_00000.parquet through 001_00009.parquet.

2. Download the tokenizer files and config.json from:
   https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2
   Model weights are not required.

3. Set the Parquet and tokenizer input paths and the three output paths
   at the top of prepare_fineweb_edu.py.

4. Run python prepare_fineweb_edu.py from this directory. The script requires
   NumPy, PyArrow, tqdm, and Transformers. Place the generated train.bin,
   val.bin, and meta.pkl in this directory if written elsewhere.
