from __future__ import print_function, division


import os
import glob
import re
import pickle
import csv
from itertools import groupby

import pandas as pd

from pymatgen import Structure
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
    return [[dtype(x) for x in line.split()] for line in lines]

def read_outcar_density_matrix(f_outcar):
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

def read_density_matrices(rundirs = None,
                          datafile = 'data/density_matrices.pickle'):
    if os.path.exists(datafile):
        print("Loading density matrices from pickle...")
        with open(datafile, 'rb') as f:
            DMs = pickle.load(f)
        return DMs

    print("Parsing density matrices and storing to pickle...")

    if not rundirs:
        basedir = "/home/adler/work/lowest_energy_comparison"
        rundirs = glob.glob(os.path.join(basedir, "Ispin*/*adler"))

    DMs = []
    for dirname in rundirs:
        vasprun = os.path.join(dirname, 'vasprun.xml')
        outcar = os.path.join(dirname, 'OUTCAR')
        try:
            print(vasprun)
            v = Vasprun(vasprun)
            name, U, spin = params_from_dir(outcar)
            data = {'NAME': name, 'U': U, 'SPIN': spin,
                    'CONVERGED': v.converged,
                    'DATA': read_outcar_density_matrix(outcar)}
            DMs.append(data)
        except:
            print("Failed to parse file", vasprun)

    with open(datafile, 'wb') as f:
        pickle.dump(DMs, f)

    return DMs

def read_consolidated_poscars(filename):
    # read gatheredPOSCARS from USPEX; return list of Structures
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
    # read list of individual POSCAR files; return list of Structures
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
