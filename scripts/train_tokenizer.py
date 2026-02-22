import argparse
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def main():
    parser = argparse.ArgumentParser(description="Train a ByteLevel BPE tokenizer.")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", required=True, help="Output tokenizer.json path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
    )
    tokenizer.train([args.corpus], trainer)
    tokenizer.save(args.out)
    print(f"Saved tokenizer to {args.out} (vocab_size={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
