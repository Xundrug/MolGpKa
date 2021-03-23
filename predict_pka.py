#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import unicode_literals
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

import h5py
import json
import os
import os.path as osp
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool

from utils.ionization_group import get_ionization_aid
from utils.descriptor import mol2vec
from net import GCNNet

root = os.path.abspath(os.path.dirname(__file__))

def load_model(model_file, device="cpu"):
    model= GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model

def model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def predict_acid(mol):
    model_file = os.path.join(root, "models/weight_acid.pth")
    model_acid = load_model(model_file)

    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid)
        acid_res.update({aid:apka})
    return acid_res

def predict_base(mol):
    model_file = os.path.join(root, "models/weight_base.pth")
    model_base = load_model(model_file)

    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base) 
        base_res.update({aid:bpka})
    return base_res

def predict(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    return base_dict, acid_dict

if __name__=="__main__":
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    base_dict, acid_dict = predict(mol)
    print("base:",base_dict)
    print("acid:",acid_dict)

