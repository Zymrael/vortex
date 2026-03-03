import argparse
import csv
from pathlib import Path
from typing import List, Union
import yaml
import numpy as np

import torch

from vortex.model.model import StripedHyena
from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
from vortex.model.utils import dotdict, load_checkpoint

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


def test_dna_model(model, device="cuda:0"):
    """
    Test a DNA sequence model by comparing its predictions to the input sequence.
    Accuracy scores are the % correctly predicted tokens, we expect >80% for these prompts.
    
    Args:
        model_name (str): Name of the model to test
        device (str): Device to run the model on (default: 'cuda:0')

    Returns:
        tuple: (original sequence, predicted sequence, accuracy)
    """
    
    sequences = read_prompts('./test/data/prompts.csv')
    losses = []
    accuracies = []

    for seq in sequences:
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
        
        # Store metrics
        losses.append(loss.cpu().item())
        accuracies.append(accuracy)

    # Print all sequence results at the end
    print("\nSequence Results:")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Sequence {i+1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
        if acc < 0.5:
            print("WARNING: Forward pass accuracy is below 50% on a test sequence. Model loading may be broken, fully trained models should have >90% accuracy.")
    
    return accuracies, losses


if __name__ == "__main__":
    """
    Test setup correctness by doing a teacher forced forward pass on a conserved gene.
    Make sure you are using the correct config and checkpoint for the model you are testing.

    python ./test/generation/test_generation.py --config_path <config_path> --checkpoint_path <path.pt>

    Expected results:

    Evo 2 40B
    Mean Loss: 0.216
    Mean Accuracy: 91.673%

    Evo 2 7B
    Mean Loss: 0.348
    Mean Accuracy: 86.346%
    
    Evo 2 1B base
    Mean Loss: 0.502
    Mean Accuracy: 79.556%
    
    """
    parser = argparse.ArgumentParser(description="Run StripedHyena Model")
    parser.add_argument(
        "--config_path", required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint_path", default=None, help="Path to checkpoint file"
    )

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args = parser.parse_args()
    assert any(model in path for model in ['evo2-40b-1m', 'evo2-20b-1m', 'evo2-7b-1m', 'evo2-1b-8k'] 
            for path in [args.config_path]), "Testing is only for Evo 2 40B, 20B, 7B and 1B"

    config = dotdict(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)
    
    m = StripedHyena(config)

    load_checkpoint(m, checkpoint_path=args.checkpoint_path)

    accuracies, losses = test_dna_model(m)
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)
    print(f"\nMean Loss: {mean_loss:.3f}")
    print(f"Mean Accuracy: {mean_accuracy * 100:.3f}%\n")

    eps = 1e-3 # epsilon for float comparison
    passed = True
    if 'evo2-40b-1m' in args.config_path:
        if not abs(mean_loss - 0.2159424) < eps: 
            print(f"Test Failed: Expected loss of 0.21594, got {mean_loss}")
            passed = False
    elif 'evo2-7b-1m' in args.config_path:
        if not abs(mean_loss - 0.3476563) < eps: 
            print(f"Test Failed: Expected loss of 0.34766, got {mean_loss}")
            passed = False
    elif 'evo2-1b-8k' in args.config_path:
        if not abs(mean_loss - 0.501953125) < eps:
            print(f"Test Failed: Expected loss of 0.50195, got {mean_loss}")
            passed = False
    else:
        print(f"Test Failed: Unexpected model config path {args.config_path}")
        passed = False

    if passed:
        print("Test Passed!")