#!/usr/bin/env python

from __future__ import division, print_function

import os
import glob
import re
import pickle
import csv
from itertools import groupby

import numpy as np
from pymatgen.io.vasp import Vasprun


def read_E_vs_U(filename = "data/E-vs-U-all-USPEX-Ran.dat"):
    '''
    Returned data structure is dict with keys = EA___ and
    values = {U: [U1, U2, ...], E: [U1, U2, ...]}
    '''
    with open(filename, "r") as csvfile:
        f = csv.reader(csvfile)
        data = [line for line in f]

    ID, SYSTEM, ENE, SPIN, UMINUSJ, KPTS = range(6)
    NATOMS = 8
    EV_TO_MEV = 1000

    keyfunc = lambda x: x[SYSTEM]
    # ignore header line + only keep spin-polarized calculations
    data = sorted([x for x in data[1:] if x[SPIN] == "True"], key = keyfunc)
    processed = {}
    for system,g in groupby(data, keyfunc):
        rows = sorted(list(g), key = lambda x: x[UMINUSJ])
        processed[system] = {
            "U": [float(row[UMINUSJ]) for row in rows],
            "E": [float(row[ENE])*EV_TO_MEV/NATOMS for row in rows],
            }
    return processed

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
    datafile = 'data/density_matrices.pickle'

    if os.path.exists(datafile):
        print("Loading density matrices from pickle...")
        with open(datafile, 'rb') as f:
            DMs = pickle.load(f)
    else:
        print("Parsing density matrices and storing to pickle...")
        DMs = get_data()
        with open(datafile, 'wb') as f:
            pickle.dump(DMs, f)

    # compute sum_spin tr{n-n^2}
    tr_rho = []
    for dm in DMs:
        if dm['U'] > 0:
            dm1 = dm['DATA'][-1]['DATA'][2]['DM1']
            dm2 = dm['DATA'][-1]['DATA'][2]['DM2']
            nn2 = np.trace(dm1 - np.dot(dm1, dm1)) + np.trace(dm2 - np.dot(dm2, dm2))
            tr_rho.append((dm['NAME'], dm['U'], nn2))

    E_vs_U = read_E_vs_U()

    import matplotlib.pyplot as plt

    # Scatter plot of tr{n-n^2} vs. U
    plt.figure()
    for u0 in [1.0, 2.0, 3.0]:
        yy = sorted([c for n,u,c in tr_rho if u == u0])
        plt.plot([u0]*len(yy), yy, 'o')
    plt.xlim(0, 4)
    plt.xlabel("U (eV)")
    plt.ylabel("$\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$")
    plt.title("Itineracy vs. U")
    plt.savefig("results/trnn2-vs-U.pdf", bbox_inches = 'tight')

    # Scatter plot of E(1)-E(0) vs. tr{n-n^2}
    plt.figure()
    trnn2 = dict((n, c) for n,u,c in tr_rho if u == 1.0)
    dE = dict((n, E_vs_U[n]['E'][1] - E_vs_U[n]['E'][0]) for n in trnn2)
    names = trnn2.keys()
    MEV_IN_EV = 1000
    plt.plot([trnn2[n] for n in names], [dE[n]/MEV_IN_EV for n in names], 'o',
             color = 'blue', alpha = 0.5)
    plt.xlabel('$\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
    plt.ylabel('dE/dU (per atom)')
    plt.title('Energy Change vs. Itineracy')
    plt.savefig('results/dE-vs-trnn2.pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    main()
