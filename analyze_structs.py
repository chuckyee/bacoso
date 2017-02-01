#!/usr/bin/env python


from __future__ import print_function, division


import glob

from pymatgen import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def main():
    files = glob.glob('*.vasp')
    structs = []
    for f in files:
        s = Structure.from_str(''.join(open(f).readlines()), fmt='poscar')
        s.filename = f          # add "filename" data member to class
        structs.append(s)

    sm = StructureMatcher()
    grouped_structs = sm.group_structures(structs)
    for struct_list in grouped_structs:
        print(len(struct_list), end = '')
        print([struct.filename for struct in struct_list])

if __name__ == '__main__':
    main()
