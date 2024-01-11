# MolGpKa
Fast and accurate prediction of the pKa values of small molecules is important in the drug discovery process since the ionization state of a drug has significant influence on its activity and ADME-Tox properties. MolGpKa is a tool for pKa prediction using graph-convolutional neural network model. The model works by learning pKa related chemical patterns automatically and building reliable predictors with learned features.

## Addendum installation
See the Docker_README.md file

## Requirements

* Python 3.6
* Pytorch >=1.4
* Pytorch-geometric (https://github.com/rusty1s/pytorch_geometric)
* RDKit (http://www.rdkit.org/docs/Install.html)
* py3Dmol 
* sklearn 0.21.3
* numpy 1.18.1
* pandas 0.25.3
* pickle

## Usage

### Using trained model for pKa prediction
`example.ipynb` is an example notebook for using MolGpKa, model weights file are located in `models`.

### Training model for convolutional-graph neural networks

1. `prepare_dataset_graph.py`--First, you should prepare the molecular file `mols.sdf` from ChEMBL database like the example. Then you will get two files `train.pickle, valid.pickle` in `datasets/` when you run the script  for data preparation.

2. `train_graph.py`--The purpose of this code is to train the graph-convolutional neural network model for pka prediction, the parameter file of MolGpKa will save in `models/`. You need to train the model for acidic ioniable center and basic ioniable center separately with corresponding data.

### Training model for AP-DNN

```
src/baseline/prepare_dataset_ap.py
src/baseline/train_ap.py
```
These scripts are designed to construct AP-DNN model which contain data preparation and model training.


## Benchmark set for pka substitution effects

In order to test the substitution effects extensively, we created a benchmark set by performing matched molecular pair analysis on experimental pKa data sets collected by Baltruschat et al. The benchmark set contains 4322 data points.


