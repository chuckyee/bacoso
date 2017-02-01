#!/usr/bin/env python

from __future__ import division, print_function

import os
import glob
import re
import pickle

import numpy as np
from pymatgen.io.vasp import Vasprun


def params_from_dir(d):
    spin, name, U = [e.split('_')[-1] for e in d.split('/')[-3].split('#')]
    return name, float(U), int(spin)

def parse_matrix(lines, dtype=float):
    return np.asarray([[dtype(x) for x in line.split()] for line in lines])

def read_density_matrix(f_outcar):
    # data structure: list of dictionaries each representing one iteration
    # {'IONSTEP': 1, 'ELSTEP': 1, 'DM': <> }
    with open(f_outcar, 'r') as f:
        lines = f.readlines()
    data = []
    iIter = [i for i,x in enumerate(lines) if "Iteration" in x]
    for i,j in zip(iIter, iIter[1:] + [-1]):
        idata = lines[i:j]
        g = re.match(r'.*Iteration(.*)\((.*)\).*', idata[0])
        dataIter = dict(
            zip(['IONSTEP', 'ELSTEP'], [int(x) for x in g.groups()]) )
        iAtom = [m for m,x in enumerate(idata) if x.startswith('atom =')]
        dataIter['DATA'] = []
        for m,n in zip(iAtom, iAtom[1:] + [-1]):
            adata = idata[m:n]
            g = re.match(r'.*atom =(.*)type =(.*)l =(.*).*', adata[0])
            dataAtom = dict(
                zip(['ATOM', 'TYPE', 'L'], [int(x) for x in g.groups()]) )
            iDM = [p for p,x in enumerate(adata) if x.startswith('spin') or
                   x.startswith(' occupancies')]
            dataAtom['DM1'] = parse_matrix(adata[iDM[0]+2:iDM[1]-1])
            dataAtom['DM2'] = parse_matrix(adata[iDM[1]+2:iDM[2]-1])
            dataAtom['OCC'] = None # not implemented
            dataAtom['VEC'] = None # not implemented
            dataIter['DATA'].append(dataAtom)
        data.append(dataIter)
    return data

def get_data():
    basedir = "/home/adler/work/lowest_energy_comparison"
    dirlist = glob.glob(os.path.join(basedir, "Ispin*/*adler"))

    DMs = []
    for dirname in dirlist:
        vasprun = os.path.join(dirname, 'vasprun.xml')
        outcar = os.path.join(dirname, 'OUTCAR')
        try:
            print(vasprun)
            v = Vasprun(vasprun)
            name, U, spin = params_from_dir(outcar)
            data = {'NAME': name, 'U': U, 'SPIN': spin,
                    'CONVERGED': v.converged,
                    'DATA': read_density_matrix(outcar)}
            DMs.append(data)
        except:
            print("Failed to parse file", vasprun)
    return DMs

def main():
    datafile = 'density_matrices.pickle'

    if os.path.exists(datafile):
        with open(datafile, 'rb') as f:
            DMs = pickle.load(f)
    else:
        DMs = get_data()
        with open(datafile, 'wb') as f:
            pickle.dump(DMs, f)

    tr_rho = []
    for dm in DMs:
        if dm['U'] > 0:
            dm1 = dm['DATA'][-1]['DATA'][2]['DM1']
            dm2 = dm['DATA'][-1]['DATA'][2]['DM2']
            nn2 = np.trace(dm1 - np.dot(dm1, dm1)) + np.trace(dm2 - np.dot(dm2, dm2))
            tr_rho.append((dm['NAME'], dm['U'], nn2))

    # don't import matplotlib unless absolutely necessary
    import matplotlib.pyplot as plt
    for u0 in [1.0, 2.0, 3.0]:
        yy = sorted([c for n,u,c in tr_rho if u == u0])
        plt.plot([u0]*len(yy), yy, 'o')
    plt.xlim(0, 4)
    plt.show()

if __name__ == '__main__':
    main()
