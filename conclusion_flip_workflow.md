1. have two trained model trajectories (either on base or base-64-subset)
2. run compute_novel_acc_full_for_set_of_snapshots for each of the trajectories
    The result (performance on novel_large) will be logged in the respective trajectory's runs folder
    novel_acc_variance_552.txt.
3. Identify a pair of algorithm snapshot (one from each trajectory) whose novel_large
    difference is around say 0.5%.
4. run compute_novel_acc_variance_PN.sh for each of these two found snapshots
    on novel large to more accurately estimate the true performance difference.
    Make sure they are actually differing by that much of a difference.
5. run compute_novel_acc_variance_PN_pair.sh to check for conclusion flip.
    Can check when difference number of subset classes are chosen, e.g.
    20, 40, 80, 160, etc.