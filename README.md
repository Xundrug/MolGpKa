# MolGpKa
Fast and accurate prediction of the pKa values of small molecules is important in the drug discovery process since the ionization state of a drug has significant influence on its activity and ADME-Tox properties. MolGpKa is a web server for pKa prediction using graph-convolutional neural network model. The model works by learning pKa related chemical patterns automatically and building reliable predictors with learned features.

## Requirements

* Python 3.6
* Pytorch 1.4
* Pytorch-geometric (https://github.com/rusty1s/pytorch_geometric)
* RDKit (http://www.rdkit.org/docs/Install.html)
* sklearn 0.21.3
* numpy 1.18.1
* pandas 0.25.3
* pickle

## Usage

`example.ipynb` is a example scripts of using MolGpKa. We use 3dmol.js to visualize the calculation results.

This is a example file format for model training, validation and test, including `idx` and `acd_pka`. `idx` is the atom id for the ionizable groupcenter, `acd_pka` is the calculation pka value from ChEMBL database.
```
datasets/mols.sdf
```
This tool only support command line. First, you should prepare the molecular file `mols.sdf` from ChEMBL database like the example. Then you will get two files `train.pickle, valid.pickle` in `datasets` when you run the script for data preparation.
```
prepare_dataset_graph.py
```
The purpose of this code is to train the graph-convolutional neural network model for pka prediction, the file of weight will save in `models`. You need to train the model for acidic ioniable center and basic ioniable center separately with corresponding data.
```
train_graph.py
```
This script contains the model architecture for GCN, GAT and MPNN. If you want to train different graph neural network, you just need to replace the model name in `train_graph.py`.
```
net.py
```
These scripts are designed to construct AP-DNN model which contain data preparation and model training.
```
prepare_dataset_ap.py
train_ap.py
```

## Benchmark set for pka substitution effects

We combined some pKa experimental data sets, and then it was processde by mmpdb for matched molecule pair analysis. Finally, We found 2910 pairs of molecules related to the substitution effects, it was listed in `benchmark_delta_pka/experimental_substituents_effects.xlsx`. Then, we filtered case with less than five substituents, the result was shown in `benchmark_delta_pka/experimental_substituents_effects_filter.xlsx`.


