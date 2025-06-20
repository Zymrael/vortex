import argparse
import csv
from typing import List, Optional, Union
from pathlib import Path
import numpy as np

import yaml


def read_prompts(
    input_file: Path,
) -> Union[List[List[str]]]:
    """Read prompts from input file."""
    promptseqs: List[str] = []

    with open(input_file, encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            promptseqs.append(row[0])

    return promptseqs


def mid_point_split(*, seq, num_tokens):
    mid_point = 2 * (len(seq) // 4)
    prompt = seq[:mid_point]
    target = seq[mid_point : mid_point + num_tokens]  # Only compare to the section of sequence directly
    return prompt, target


def generate_and_score(*, sequences, model, tokenizer, args, generations_per_prompt=5, device="cuda:0"):
    """
    Prompt with first half, generate and score on 2nd half
    """
    import torch
    from vortex.model.generation import generate

    scores = []
    prompts = []
    targets = []

    # Prepare all prompts and targets
    for seq in sequences:
        prompt, target = mid_point_split(seq=seq, num_tokens=args.num_tokens)

        # Repeat prompt for multiple generations
        prompts.extend([prompt] * generations_per_prompt)
        targets.extend([target] * generations_per_prompt)

    for i in range(len(prompts)):
        prompt = prompts[i]
        target = targets[i]

        with torch.inference_mode():

            ret = generate(
                prompt_seqs=[prompt],
                n_tokens=args.num_tokens,
                model=model,
                tokenizer=tokenizer,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                device=device,
                verbose=1,
                cached_generation=args.cached_generation,
            )

            decoded_seq = ret.sequences[0]
            score = calculate_sequence_identity(decoded_seq, target)
            scores.append(score)
    
    return scores


def calculate_sequence_identity(seq1: str, seq2: str) -> Optional[float]:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None

    # Direct comparison of sequences
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))

    return (matches / min_length) * 100


def main():
    '''
    Test setup cortectness by greedily generating sequences from prompts and comparing to the target sequences. 
    python ./test/generation/test_generation.py --config_path <config_path> --checkpoint_path <path.pt>

    Expected results (direct comparison of sequences without alignment):
    Evo 2 40B 1m: 91.15%
    Evo 2 7B 1m: 89.25% 
    Evo 2 1B base: 68.0%
    '''
    import torch

    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
    from vortex.model.utils import dotdict, load_checkpoint

    parser = argparse.ArgumentParser(description="Run StripedHyena Model")
    parser.add_argument("--config_path", required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint file")
    parser.add_argument(
        "--cached_generation",
        action="store_true",
        help="Use caching to speed up generation.",
    )

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args = parser.parse_args()
    assert any(model in path for model in ['evo2-40b-1m', 'evo2-7b-1m', 'evo2-1b-8k'] 
            for path in [args.config_path]), "Testing is only for Evo 2 40B, Evo 2 7B, and Evo 2 1B base"

    args.num_tokens = 500
    args.temperature = 1.0
    args.top_k = 1
    args.top_p = 1.0
    args.generations_per_prompt = 1

    config = dotdict(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = StripedHyena(config)

    load_checkpoint(m, checkpoint_path=args.checkpoint_path)

    sequences = read_prompts("./test/data/prompts.csv")

    scores = generate_and_score(
        sequences=sequences,
        model=m,
        tokenizer=tokenizer,
        args=args,
        generations_per_prompt=args.generations_per_prompt
    )

    print(scores)
    mean_score = np.mean(scores)
    print("% Matching Nucleotides")
    print(mean_score)

    eps = 3 # Increased epsilon, different results from different environments due to numerical differences that cause token changes
    passed = True
    if 'evo2-40b-1m' in args.config_path:
        assert mean_score - 91.15 < eps, f"Test Failed: Expected mean score of 91.15, got {mean_score}"
    elif 'evo2-7b-1m' in args.config_path:
        assert mean_score - 89.25 < eps, f"Test Failed: Expected mean score of 89.25, got {mean_score}"
    elif 'evo2-1b-8k' in args.config_path:
        assert mean_score - 68.0 < eps, f"Test Failed: Expected mean score of 68.0, got {mean_score}"
    else:
        print(f"Test Failed: Did not recognize config path {args.config_path}")
        print("Test only supports Evo 2 40B, Evo 2 7B, or Evo 2 1B base")
        passed = False

    if passed:
        print("Test Passed")
    else:
        print("Test Failed")

    


if __name__ == "__main__":
    main()
