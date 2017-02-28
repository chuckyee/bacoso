#!/usr/bin/env python

from __future__ import division, print_function

import os
from monty.serialization import loadfn

import pandas as pd
from pymatgen.io.vasp import sets

import ios


class BCSOSet(sets.DictSet):
    """
    Customized implementation of VaspInputSet utilizing parameters set locally
    in file BCSOSet.yaml, which is based on MPRelaxSet.yaml.
    """
    CONFIG = loadfn(os.path.join("BCSOSet.yaml"))

    def __init__(self, structure, **kwargs):
        super(BCSOSet, self).__init__(
            structure, BCSOSet.CONFIG, **kwargs)
        self.kwargs = kwargs



def main():
    structs = ios.read_consolidated_poscars('data/gatheredPOSCARS')
    energies = pd.read_csv("data/E-vs-U-all-USPEX-Ran.dat")

    # get list of low-energy structs
    low_energy_ids = energies[energies.U_minus_J == 3].SYSTEM.values
    low_energy_structs = [struct for struct in structs
                          if struct.name in low_energy_ids]

    # write VASP inputs for each structure
    base_dir = 'inputs'
    for struct in low_energy_structs:
        v = BCSOSet(struct)
        output_dir = os.path.join(base_dir, struct.name)
        v.write_input(output_dir, make_dir_if_not_present=True)

if __name__ == '__main__':
    main()
