import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from multiprocessing import Pool

APLength = 6

def calc_ap(mol):
    aid = int(mol.GetProp("idx"))
    pka = float(mol.GetProp("pka")) 
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, maxLength=APLength, fromAtoms=[aid])
    arr = np.zeros(1, )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, pka

def generate_datasets(mols):
    fps, targets= [], []
    for m in mols:
        if not m:
            continue
        fp, pka = calc_ap(m)
        fps.append(fp)
        targets.append(pka)
    fps = np.asarray(fps)
    targets = np.asarray(targets)
    path = "datasets/datasets_ap.npz"
    np.savez_compressed(path, fp=fps, pka=targets)
    return

if __name__=="__main__":
    mol_path = "datasets/mols.sdf"
    mols = Chem.SDMolSupplier(mol_path, removeHs=False)
    generate_datasets(mols)

