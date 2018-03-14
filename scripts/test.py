#!/usr/bin/env python
# encoding: utf-8
# Created Time: 六  3/ 3 09:12:41 2018

import os
import logging

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer_chemistry.dataset.preprocessors import SchNetPreprocessor
from chainer_chemistry.datasets import qm9
from foundation.utils import rdkit_utils
from foundation.utils import numpy_utils

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit import RDLogger



def get_dist(sm):
    mol = Chem.MolFromSmiles(sm)
    confid = AllChem.EmbedMolecule(mol)
    dist_matrix = rdmolops.Get3DDistanceMatrix(mol, confId=confid)

    return dist_matrix

def parse_xyz(xyz_file):
    with open(xyz_file, 'rb') as f:
        data = [line.strip() for line in f]

    num_atom = int(data[0])
    properties = list(map(float, data[1].split('\t')[1:]))
    smiles = data[3 + num_atom].split('\t')
    coords = [x.split('\t')[1:4] for x in data[2: 2+num_atom]]
    coords = np.asarray(coords, dtype=np.float32)

    return {'num_atom': num_atom,
            'properties': properties,
            'SMILES1': smiles[0],
            'SMILES2': smiles[1],
            'coords': coords}

def main():

    xyz_file = '/Users/zhaohuizhu/datasets/mol_data/qm9/xyz/dsgdb9nsd_089262.xyz'

    preprocessor = SchNetPreprocessor()
    data = qm9.get_qm9(preprocessor)
    print(len(data._datasets[0]))
    print(len(data._datasets[1]))

    print([x.shape for x in data._datasets[0][-10:]])
    print([x.shape for x in data._datasets[1][-10:]])


if __name__ == '__main__':
    main()
