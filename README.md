# NanoTransformer

Stage-wise implementation of a tiny causal LM (context_len=3, ~2.1M params).

## Stage 1: Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Stage 2: Tokenizer and data
1) Download a 50MB English corpus (optional helper):
```bash
python scripts/download_corpus.py --out data/corpus.txt --target-mb 50
```
2) Or place your UTF-8 corpus at data/corpus.txt
3) Train BPE tokenizer:
```bash
python scripts/train_tokenizer.py --corpus data/corpus.txt --vocab-size 10000 --out artifacts/tokenizer.json
```
4) Build token ids and splits:
```bash
python scripts/prepare_data.py --corpus data/corpus.txt --tokenizer artifacts/tokenizer.json --out artifacts/data_ids.pt --splits-out artifacts/splits.pt
```

## Stage 3: Train
```bash
python train.py --config configs/base.json
```

## Stage 4: Evaluate
```bash
python eval.py --config configs/base.json --checkpoint checkpoints/best.pt
```

## Stage 5: Trigram baseline (add-k smoothing)
```bash
python baseline_trigram.py --config configs/base.json --k 0.1
```

## Stage 6: Latency
```bash
python measure_latency.py --config configs/base.json --checkpoint checkpoints/best.pt --iters 200 --warmup 10
```

## Notes
- Adjust hyperparameters in configs/base.json
- Outputs: artifacts/ (tokenizer, data, metrics) and checkpoints/ (model)
