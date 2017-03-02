#!/usr/bin/env python

from __future__ import division, print_function

import os
import shutil
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

submit_file = """#!/bin/sh

#$ -N {}
#$ -pe mpi2_14_one 12
#$ -q wp12
#$ -j y
#$ -M chuckyee@physics.rutgers.edu
#$ -m e
#$ -v LD_LIBRARY_PATH

# do NOT remove the following line!
source $TMPDIR/sge_init.sh

source ~/.bashrc
export SMPD_OPTION_NO_DYNAMIC_HOSTS=1
export OMP_NUM_THREADS=1

BIN=vasp

python custvasp.py /opt/mpich2/intel/14.0/bin/mpiexec -n $NSLOTS -machinefile $TMPDIR/machines -port $port $BIN
"""

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
        with open(os.path.join(output_dir, 'wp12.vasp'), 'w') as f:
            f.write(submit_file.format(struct.name))
        shutil.copy('custvasp.py', output_dir)

if __name__ == '__main__':
    main()
