from utils.descriptor import mol2vec
import pickle
from sklearn.model_selection import train_test_split

from rdkit import Chem


def dump_datasets(dataset, path):
    dataset_dumps = pickle.dumps(dataset)
    with open(path, "wb") as f:
        f.write(dataset_dumps)
    return

def generate_datasets(mols):
    datasets = []
    for mol in mols:
        if not mol:
            continue
        atom_idx = int(mol.GetProp("idx"))
        pka = float(mol.GetProp("pka"))
        data = mol2vec(mol, atom_idx, evaluation=False, pka=pka)
        datasets.append(data)
    
    train_dataset, valid_dataset = train_test_split(datasets, test_size=0.1)
    return train_dataset, valid_dataset

if __name__=="__main__":
    mol_path = "datasets/mols.sdf"
    mols = Chem.SDMolSupplier(mol_path, removeHs=False)
    train_dataset, valid_dataset = generate_datasets(mols)

    train_path = "datasets/train.pickle"
    valid_path = "datasets/valid.pickle"
    dump_datasets(train_dataset, train_path)
    dump_datasets(valid_dataset, valid_path)
