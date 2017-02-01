#!/usr/bin/env python


from __future__ import print_function, division


import re
import glob
import pandas as pd
from pymatgen import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def read_consolidated_poscars(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    ii = [i for i,line in enumerate(lines) if line.startswith('EA')]
    structs = []
    for i,j in zip(ii, ii[1:]+[None]):
        s = Structure.from_str(''.join(lines[i:j]), fmt = 'poscar')
        name, a, b, c, alpha, beta, gamma, _, sgroup = lines[i].split()
        s.name = name
        structs.append(s)
    return structs

def read_poscar_files(filenames):
    structs = []
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
        s = Structure.from_str(''.join(lines), fmt='poscar')
        s.name = f
        structs.append(s)
    return structs

def read_individuals_file(filename):
    # read Individuals file from USPEX run to sort candidates by enthalpy
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[2:]:      # ignore labels
        matches = re.match(r'(.*)\[(.*)\](.*)\[(.*)\](.*)', line)
        head, comp, mid, kpts, tail = matches.groups()
        row = head.split() + [[int(n) for n in comp.split()]] + mid.split() +\
              [[int(n) for n in kpts.split()]] + tail.split()
        data.append(row)
    labels = ('Gen   ID    Origin   Composition    Enthalpy   Volume  Density'
              '   Fitness   KPOINTS  SYMM  Q_entr A_order S_order')
    IDs = [row[1] for row in data]
    return pd.DataFrame(data, index = IDs, columns = labels.split())

def main():
    # read_poscar_files(glob.glob('*.vasp'))
    structs = read_consolidated_poscars('USPEX-structures/gatheredPOSCARS')
    df = read_individuals_file('USPEX-structures/Individuals')
    print(df)

    # sm = StructureMatcher()
    # grouped_structs = sm.group_structures(structs)
    # for struct_list in grouped_structs:
    #     print(len(struct_list), end = '')
    #     print([struct.name for struct in struct_list])

if __name__ == '__main__':
    main()
