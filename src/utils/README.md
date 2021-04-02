1. `smarts_pattern.tsv`--it list smarts patterns for matching ionizable center, including SMARTS, ionizable group type and ionizable center id.

2. `ionization_group.py`--It's script for matching ionizable center by smarts pattern.

3. `descriptor.py`--It's a script for calculating molecular graph by RDKit to train and predict pKa.

4. `net.py`--This script contains the model architecture for GCN, GAT and MPNN. If you want to train different graph neural network, you just need to replace the model name in `train_graph.py`.