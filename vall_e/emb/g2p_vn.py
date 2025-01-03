import argparse
import random
from pathlib import Path
import re

from viphoneme import vi2IPA
import torch
from tqdm import tqdm


def _get_graphs(path):
    """Read the content of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        graphs = f.read()
    return graphs


def encode(graphs: str) -> str:
    """
    Convert Vietnamese text into IPA phonemes using vi2IPA.

    Args:
        graphs (str): The input text.
    Returns:
        str: The IPA representation of the input text.
    """
    # Remove non-alphanumeric characters except Vietnamese characters
    pattern = '[^a-zA-ZÀ-ỹ\s]+'
    graphs = re.sub(pattern, ' ', graphs).strip()
    ipa = vi2IPA(graphs)
    return ipa


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder containing text files")
    parser.add_argument("--suffix", type=str, default=".txt", help="Suffix of text files to process")
    parser.add_argument("--out_dir", type=Path, help="Output directory for phoneme files")
    args = parser.parse_args()

    # Collect all files with the specified suffix
    paths = list(args.folder.rglob(f"*{args.suffix}"))
    random.shuffle(paths)

    # Create the output directory if specified
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        
    for path in tqdm(paths):
        # Determine the output path for the phoneme file
        if args.out_dir:
            phone_path = args.out_dir / path.relative_to(args.folder).with_name(path.stem + ".phn.txt")
            phone_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            phone_path = path.with_name(path.stem + ".phn.txt")

        # Skip if the output file already exists
        if phone_path.exists():
            continue

        # Process the file
        graphs = _get_graphs(path)
        try:
            phones = encode(graphs)
            with open(phone_path, "w", encoding="utf-8") as f:
                f.write(phones)
        except Exception as e:
            print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    main()
