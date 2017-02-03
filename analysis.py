#!/usr/bin/env python

from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.analysis.structure_matcher import StructureMatcher

import ios


def analyze_trnn2():
    DMs = ios.read_density_matrices()
    E_vs_U = ios.read_E_vs_U()

    # compute sum_spin tr{n-n^2}
    trnn2 = []
    for dm in DMs:
        if dm['U'] > 0:
            dm1 = dm['DATA'][-1]['DATA'][2]['DM1']
            dm2 = dm['DATA'][-1]['DATA'][2]['DM2']
            nn2 = np.trace(dm1 - np.dot(dm1, dm1)) + np.trace(dm2 - np.dot(dm2, dm2))
            trnn2.append((dm['NAME'], dm['U'], nn2))

    # Scatter plot of tr{n-n^2} vs. U
    plt.figure()
    for u0 in [1.0, 2.0, 3.0]:
        yy = sorted([c for n,u,c in trnn2 if u == u0])
        plt.plot([u0]*len(yy), yy, 'o')
    plt.xlim(0, 4)
    plt.xlabel("U (eV)")
    plt.ylabel("$\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$")
    plt.title("Itineracy vs. U")
    plt.savefig("results/trnn2-vs-U.pdf", bbox_inches = 'tight')

    # Scatter plot of E(1)-E(0) vs. tr{n-n^2}
    plt.figure()
    trnn2u1 = dict((n, c) for n,u,c in trnn2 if u == 1.0)
    dE = dict((n, E_vs_U[n]['E'][1] - E_vs_U[n]['E'][0]) for n in trnn2u1)
    names = trnn2u1.keys()
    MEV_IN_EV = 1000
    plt.plot([trnn2u1[n] for n in names], [dE[n]/MEV_IN_EV for n in names],
             'o', color = 'blue', alpha = 0.5)
    plt.xlabel('$\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
    plt.ylabel('dE/dU (per atom)')
    plt.title('Energy Change vs. Itineracy')
    plt.savefig('results/dE-vs-trnn2.pdf', bbox_inches = 'tight')


def analyze_low_e_structs():
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
    # analyze_trnn2()
    analyze_low_e_structs()
