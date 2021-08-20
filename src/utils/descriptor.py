from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops

import numpy as np
from utils.ionization_group import get_ionization_aid

import torch
from torch_geometric.data import Data

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def get_atom_features(mol, aid):
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    acceptor_smarts_one = '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
    acceptor_smarts_two = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    donor_smarts_one = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
    donor_smarts_two = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

    hydrogen_donor_one = Chem.MolFromSmarts(donor_smarts_one)
    hydrogen_donor_two = Chem.MolFromSmarts(donor_smarts_two)
    hydrogen_acceptor_one = Chem.MolFromSmarts(acceptor_smarts_one)
    hydrogen_acceptor_two = Chem.MolFromSmarts(acceptor_smarts_two)

    hydrogen_donor_match_one = mol.GetSubstructMatches(hydrogen_donor_one)
    hydrogen_donor_match_two = mol.GetSubstructMatches(hydrogen_donor_two)
    hydrogen_donor_match = []
    hydrogen_donor_match.extend(hydrogen_donor_match_one)
    hydrogen_donor_match.extend(hydrogen_donor_match_two)
    hydrogen_donor_match = list(set(hydrogen_donor_match))

    hydrogen_acceptor_match_one = mol.GetSubstructMatches(hydrogen_acceptor_one)
    hydrogen_acceptor_match_two = mol.GetSubstructMatches(hydrogen_acceptor_two)
    hydrogen_acceptor_match = []
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_one)
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_two)
    hydrogen_acceptor_match = list(set(hydrogen_acceptor_match))

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        o = []
        o += one_hot(atom.GetSymbol(), ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                        'I'])
        o += [atom.GetDegree()]
        o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                               Chem.rdchem.HybridizationType.SP2,
                                               Chem.rdchem.HybridizationType.SP3,
                                               Chem.rdchem.HybridizationType.SP3D,
                                               Chem.rdchem.HybridizationType.SP3D2])
        o += [atom.GetImplicitValence()]
        o += [atom.GetIsAromatic()]
        o += [ring.IsAtomInRingOfSize(atom_idx, 3),
              ring.IsAtomInRingOfSize(atom_idx, 4),
              ring.IsAtomInRingOfSize(atom_idx, 5),
              ring.IsAtomInRingOfSize(atom_idx, 6),
              ring.IsAtomInRingOfSize(atom_idx, 7),
              ring.IsAtomInRingOfSize(atom_idx, 8)]

        o += [atom_idx in hydrogen_donor_match]
        o += [atom_idx in hydrogen_acceptor_match]
        o += [atom.GetFormalCharge()]
        if atom_idx == aid:
            o += [0]
        else:
            o += [len(Chem.rdmolops.GetShortestPath(mol, atom_idx, aid))]

        if atom_idx == aid:
            o += [True]
        else:
            o += [False]
        m.append(o)
    return m

def mol2vec(mol, atom_idx, evaluation=True, pka=None):
    node_f = get_atom_features(mol, atom_idx)
    edge_index = get_bond_pair(mol)
    if evaluation:
        batch = np.zeros(len(node_f), )
        data = Data(x=torch.tensor(node_f, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    batch=torch.tensor(batch, dtype=torch.long))
    else:
        data = Data(x=torch.tensor(node_f, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    pka=torch.tensor([[pka]], dtype=torch.float))
    return data



