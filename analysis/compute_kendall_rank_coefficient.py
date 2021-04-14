import os
from tqdm import tqdm
import json
import argparse
import pickle
import numpy as np
import pandas as pd



def str2bool(arg):
    return arg.lower() == 'true'



def main(args):

    # read test, val accs
    print(f"Val File {args.val_file}")
    val_accs = pd.read_csv(args.val_file)["Value"].values
    print(f"Test File {args.test_file}")
    test_accs = pd.read_csv(args.test_file)["Value"].values
    
    # select epochs for evaluation
    selected_val_accs = val_accs[args.start_epoch-1:args.end_epoch]
    selected_test_accs = test_accs[args.start_epoch-1:args.end_epoch]
    assert len(selected_val_accs) == len(selected_test_accs)
    n_epochs = len(selected_val_accs)
    assert n_epochs > 1
    print(f"Selected {n_epochs} accs from validation and test accuracies, spanning {args.start_epoch}-{args.end_epoch}")

    # compute Kendall Rank Corr Coefficient
    n_positive_pairs = 0 # keep count of how many pairs of accuracy values retain order b/w val and test
    n_negative_pairs = 0 # keep count of how many pairs of accuracy values have order mismatch b/w val and test
    for i in range(n_epochs):
        for j in range(i):
            match = 0 # checks if a match has occured or not for the given pair
            if selected_val_accs[i] >= selected_val_accs[j]:
                match = int(selected_test_accs[i] >= selected_test_accs[j])
            else:
                match = int(selected_test_accs[i] < selected_test_accs[j])
            # if match is 0 then the pairs disagree, match is 1 iff pairs agree
            n_positive_pairs += match
            n_negative_pairs += (1-match)
    
    total_pairs = n_epochs * (n_epochs - 1) / 2
    assert n_positive_pairs + n_negative_pairs == total_pairs
    print(f"KRR (rho): {(n_positive_pairs-n_negative_pairs)/total_pairs}")




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--val-file', type=str, required=True)
    parser.add_argument('--test-file', type=str, required=True)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--end-epoch', type=int, default=60)
    args = parser.parse_args()
    main(args)