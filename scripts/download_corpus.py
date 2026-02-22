import argparse
import os
import shutil
import zipfile
from urllib.request import urlopen


DEFAULT_URL = (
    "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/"
    "resolve/main/wikitext-103-raw-v1.zip"
)


def download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return
    with urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def find_raw_files(root_dir):
    candidates = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.startswith("wiki.") and name.endswith(".raw"):
                candidates.append(os.path.join(root, name))

    order = {"wiki.train.raw": 0, "wiki.valid.raw": 1, "wiki.test.raw": 2}
    candidates.sort(key=lambda p: order.get(os.path.basename(p), 99))
    return candidates


def build_corpus(raw_files, out_path, target_bytes):
    written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in raw_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    b = line.encode("utf-8")
                    if written + len(b) > target_bytes:
                        return written
                    out_f.write(line)
                    written += len(b)
    return written


def main():
    parser = argparse.ArgumentParser(description="Download a 50MB English corpus.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/corpus.txt")
    parser.add_argument("--target-mb", type=int, default=50)
    parser.add_argument("--work-dir", default="data/wikitext-103-raw")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024

    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(
            f"{args.out} exists. Use --overwrite to replace it."
        )

    zip_path = os.path.join("data", "wikitext-103-raw-v1.zip")
    download(args.url, zip_path)

    os.makedirs(args.work_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.work_dir)

    raw_files = find_raw_files(args.work_dir)
    if not raw_files:
        raise FileNotFoundError("No wiki.*.raw files found after extraction.")

    written = build_corpus(raw_files, args.out, target_bytes)
    print(f"Wrote {written} bytes to {args.out}")


if __name__ == "__main__":
    main()
