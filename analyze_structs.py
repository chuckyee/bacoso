#!/usr/bin/env python

from __future__ import print_function, division

import glob
from pymatgen.analysis.structure_matcher import StructureMatcher

import ios


def main():
    # read_poscar_files(glob.glob('*.vasp'))
    structs = ios.read_consolidated_poscars('data/gatheredPOSCARS')
    df = ios.read_individuals_file('data/Individuals')

    energy_cutoff = 0.5         # eV / unit cell
    low_energy_IDs = df[df.Enthalpy < min(df.Enthalpy) + energy_cutoff].ID
    low_energy_IDs = ['EA{}'.format(id) for id in low_energy_IDs]

    low_energy_structs = [s for s in structs if s.name in low_energy_IDs]

    sm = StructureMatcher()
    grouped_low_energy_structs = sm.group_structures(low_energy_structs)

    for struct_list in grouped_low_energy_structs:
        IDs = [struct.name for struct in struct_list]
        print("{:2} | {}".format(len(struct_list), ' '.join(IDs)))

if __name__ == '__main__':
    main()
