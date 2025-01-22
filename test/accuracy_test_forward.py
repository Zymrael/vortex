

import argparse
import os
import csv
from typing import Dict, List, Optional, Tuple, Any, NamedTuple, Union
from pathlib import Path
import numpy as np
from Bio import Align
from Bio.Seq import Seq

import torch
import yaml

from vortex.model.generation import Generator
from vortex.model.model import StripedHyena
from vortex.model.sample import sample
from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
from vortex.model.utils import dotdict, print_rank_0, load_checkpoint

def test_dna_model(model, device='cuda:0'):
    """
    Test a DNA sequence model by comparing its predictions to the input sequence.
    We expect accuracy to be high for this highly conserved 16S ribosomal RNA gene.
    Scores of 25-30% suggest a broken model loading.
    
    Args:
        model_name (str): Name of the model to test
        device (str): Device to run the model on (default: 'cuda:0')
        
    Returns:
        tuple: (original sequence, predicted sequence, accuracy)
    """
    # Test sequence: E. coli BD1 16S ribosomal RNA gene, a conserved gene
    seq = "AAATTGAAGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGGAAGAAGCTTGCTCTTTGCTGACGAGTGGCGGACGGGTGAGTAATGTCTGGGAAACTGCCTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCATAACGTCGCAAGACCAAAGAGGGGGACCTTCGGGCCTCTTGCCATCGGATGTGCCCAGATGGGATTAGCTAGTAGGTGGGGTAACGGCTCACCTAGGCGACGATCCCTAGCTGGTCTGAGAGGATGACCAGCCACACTGGAACTGAGACACGGTCCAGACTCCTACGGGAGGCAGCAGTGGGGAATATTGCACAATGGGCGCAAGCCTGATGCAGCCATGCCGCGTGTATGAAGAAGGCCTTCGGGTTGTAAAGTACTTTCAGCGGGGAGGAAGGGAGTAAAGTTAATACCTTTGCTCATTGACGTTACCCGCAGAAGAAGCACCGGCTAACTCCGTGCCAGCAGCCGCGGTAATACGGAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCACGCAGGCGGTTTGTTAAGTCA"

    input_ids = torch.tensor(
        tokenizer.tokenize(seq),
        dtype=torch.int,
    ).to(device).unsqueeze(0)

    with torch.no_grad():
        output1, _ = model.forward(input_ids)
    logprobs = torch.log_softmax(output1[:, :-1, :], dim=-1)
    chars = torch.argmax(logprobs, dim=-1)

    target_ids = input_ids[:, 1:]
    pred_logits = output1[:, :-1, :]
    
    # Calculate cross entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred_logits.reshape(-1, pred_logits.size(-1)), target_ids.reshape(-1).long())
    
    # Convert predictions to sequence
    pred_tokens = [tokenizer.decode_token(s) for s in chars[0]]
    pred_seq = ''.join(pred_tokens)
    
    # Calculate accuracy
    accuracy = (target_ids[:,:] == chars[:,:]).sum().item()/(input_ids.size(1)*input_ids.size(0))
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    print("Input Sequence:")
    print(seq)
    print("\nPredicted Sequence:")
    print(pred_seq)
    print("\n Loss: {:.3}".format(loss))
    print("\nAccuracy: {:.2%}".format(accuracy))
    print("-" * 80)

    if accuracy < 0.5:
        print("WARNING: Teacher-forced accuracy is below 50%. Model loading may be broken, fully trained models should have >90% accuracy.")
    
    return seq, pred_seq, accuracy, loss.item()


if __name__ == "__main__":
    '''
    Test checkpoint correctenss by doing a teacher forced forward pass on a conserved gene.
    
    Expected accuracy values:
    Evo2 7b 500k - 98.16%

    python ./test/generation/test_generation.py --config_path <config_path> --checkpoint_path <path.pt>
    '''
    parser = argparse.ArgumentParser(description="Run StripedHyena Model")
    parser.add_argument("--config_path", required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint file")

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args = parser.parse_args()

    config = dotdict(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)
    
    m = StripedHyena(config)

    load_checkpoint(m, checkpoint_path=args.checkpoint_path)

    test_dna_model(m)