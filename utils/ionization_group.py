#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import unicode_literals
import numpy as np
from rdkit import Chem

import os
import pandas as pd

root = os.path.abspath(os.path.dirname(__file__))
smarts_file = os.path.join(root, "smarts_pattern.tsv")

def split_acid_base_pattern(smarts_file):
    df_smarts = pd.read_csv(smarts_file, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base

def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m)==1]))
    double_matches = [m for m in matches if len(m)==2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches

def match_acid(df_smarts_acid, mol):
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_acid.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        if len(index) > 2:
            index = index.split(",")
            index = [int(i) for i in index]
            for m in match:
                matches.append([m[index[0]], m[index[1]]])
        else:
            index = int(index)
            for m in match:
                matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def match_base(df_smarts_base, mol):
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        index = int(index)
        for m in match:
            matches.append([m[index]])
    matches = unique_acid_match(matches)
    matches_modify = []
    for i in matches:
        for j in i:
            matches_modify.append(j)
    return matches_modify

def get_ionization_aid(mol, acid_or_base=None):
    df_smarts_acid, df_smarts_base = split_acid_base_pattern(smarts_file)

    if mol == None:
        raise RuntimeError("read mol error: {}".format(mol_file))
    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)
    if acid_or_base == None:
        return acid_matches, base_matches
    elif acid_or_base == "acid":
        return acid_matches
    else:
        return base_matches

if __name__=="__main__":
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    matches = get_ionization_aid(mol)
    print(matches)
