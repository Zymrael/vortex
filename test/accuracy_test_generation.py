

import argparse
import os
import csv
from typing import Dict, List, Optional, Tuple, Any, NamedTuple, Union
from pathlib import Path
import numpy as np
from Bio import Align
from Bio.Seq import Seq

import yaml


def read_prompts(
    input_file: Path, 
) -> Union[List[List[str]]]:
    """Read prompts from input file."""
    promptseqs: List[str] = []
    
    with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            promptseqs.append(row[0])

    return promptseqs

def mid_point_split(*, seq, num_tokens):
    mid_point = 3*(len(seq)//4)
    prompt = seq[:mid_point]
    target = seq[mid_point:mid_point+num_tokens] #Only compare to the section of sequence directly
    return prompt, target

def generate_and_score(sequences, generator, tokenizer, args, generations_per_prompt=5, device='cuda:0'):
    """
    Prompt with first half, generate and score on 2nd half
    """
    import torch

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
            # for tokenized_prompt in tokenized_prompts:
            generated_seq = generator.generate(
                num_tokens=args.num_tokens,
                cached_generation=args.cached_generation,
                input_string=prompt,
                device=device,
                verbose=True,
                print_generation=False,
                max_seqlen=8192
            )[0].cpu().numpy()[0]
            decoded_seq = tokenizer.detokenize(generated_seq)
            score = calculate_sequence_identity(decoded_seq, target)
            scores.append(score)
    
    # Reshape scores to group by original sequence
    reshaped_scores = [scores[i:i + generations_per_prompt] 
                      for i in range(0, len(scores), generations_per_prompt)]
    
    return reshaped_scores
    

def calculate_sequence_identity(seq1: str, seq2: str, amino_acids=False) -> Optional[float]:
    """Calculate sequence identity between two sequences."""
    if not seq1 or not seq2:
        return None

    if amino_acids:
        seq1 = seq1[:len(seq1) - (len(seq1) % 3)]
        seq2 = seq2[:len(seq2) - (len(seq2) % 3)]
        aa1 = str(Seq(seq1).translate())
        aa2 = str(Seq(seq2).translate())
        seq1 = aa1
        seq2 == aa2

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    alignment = aligner.align(seq1, seq2)[0]

    matches = sum(a == b for a, b in zip(alignment[0], alignment[1]))

    return (matches / min(len(seq1),len(seq2))) * 100

def main():
    '''
    python ./test/generation/test_generation.py --config_path <config_path> --checkpoint_path <path.pt>
    '''
    import torch

    from vortex.model.generation import Generator
    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
    from vortex.model.utils import dotdict

    parser = argparse.ArgumentParser(description="Run StripedHyena Model")
    parser.add_argument("--config_path", required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint file")
    parser.add_argument("--num_tokens", default=100, help="Number of tokens to generate.")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_k", default=4, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--generations_per_prompt", default=1, type=int)
    parser.add_argument(
        "--cached_generation",
        action="store_true",
        help="Use caching to speed up generation.",
    )

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args = parser.parse_args()

    config = dotdict(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        m = StripedHyena(config).to(torch.float32)

    if args.checkpoint_path:
        m.custom_load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)

    m = m.to(device)
    m.to_bfloat16_except_pr_lc()

    g = Generator(m, tokenizer, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

    sequences = read_prompts('./test/data/prompts.csv')

    scores = generate_and_score(sequences, g, tokenizer, args, generations_per_prompt=args.generations_per_prompt)
    
    print(scores)
    print("\% Matching Nucleotides")
    print(np.mean(scores))

if __name__ == "__main__":
    main()
