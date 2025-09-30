from rdkit import Chem 
from rdkit.Chem import MolStandardize 
from typing import Any, List
import pandas as pd
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from multiprocessing  import Pool, Process

class AbstractProcessor(metaclass=ABCMeta):

    def __init__(self, rules=None):
        self.rules = rules
    
    @abstractmethod
    def preprocess() -> List[Any]:
        raise NotImplementedError

class MolPreprocessor(AbstractProcessor):

    def __init__(self, smiles: List[str]):
        super().__init__()
        self.smiles = smiles
    
    def preprocess(self, num=4, minimum_heavy_atoms=4):
        with Pool(processes=num) as p:
            args = [(smi, minimum_heavy_atoms) for smi in self.smiles]
            result = p.map(self._process, tqdm(args))
        return result
    
    def _process(self, arg):
        smi = arg[0]
        minimum_heavy_atoms = arg[1]
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            return None
        
        try:     
            mol = self._fragment_choose(mol)
        except:
            return None
        if mol.GetNumHeavyAtoms() < minimum_heavy_atoms:
            return None
        try:
            return Chem.MolToSmiles(mol)
        except:
            return None
    
    def _fragment_choose(self, mol):
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        return lfc.choose(mol)