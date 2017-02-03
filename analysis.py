#!/usr/bin/env python

from __future__ import division, print_function

import glob
import pickle
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
    # structs = ios.read_consolidated_poscars('data/gatheredPOSCARS')
    df = ios.read_individuals_file('data/Individuals')
    E_vs_U = ios.read_E_vs_U()
    with open('data/grouped-structs.pickle', 'rb') as f:
        grouped_structs = pickle.load(f)

    energy_cutoff = 0.5         # eV / unit cell
    low_energy_IDs = df[df.Enthalpy < min(df.Enthalpy) + energy_cutoff].ID
    low_energy_IDs = ['EA{}'.format(id) for id in low_energy_IDs]

    low_e_grouped_structs = [[s for s in group if s.name in low_energy_IDs]
                             for group in grouped_structs]
    low_e_grouped_structs = [group for group in low_e_grouped_structs if group]

    # low_energy_structs = [s for s in structs if s.name in low_energy_IDs]
    # sm = StructureMatcher()
    # grouped_low_energy_structs = sm.group_structures(low_energy_structs)

    # stem and leaf plot (of sorts)
    for group in low_e_grouped_structs:
        IDs = [struct.name for struct in group]
        print("{:2} | {}".format(len(group), ' '.join(IDs)))

    def compute_grouped_energies(U):
        grouped_energies = []
        for group in low_e_grouped_structs:
            IDs = [struct.name for struct in group]
            energies = [E_vs_U[ID]['E'][U] for ID in IDs if ID in E_vs_U]
            if energies:
                grouped_energies.append(energies)
        return grouped_energies

    def concatenate_groups(grouped_energies):
        xx = []
        yy = []
        for i,EE in enumerate(grouped_energies):
            xx += [i]*len(EE)
            yy += EE
        return np.asarray(xx), np.asarray(yy)

    plt.figure()
    ii = range(4)
    E0s = [min(v['E'][i] for v in E_vs_U.values()) for i in ii]
    for i in ii:
        xx, yy = concatenate_groups(sorted(compute_grouped_energies(i)))
        offset = 50 * i
        plt.plot(xx, yy - E0s[i] + offset,
                 'o', alpha = 0.5, label = 'U = {}eV'.format(i))

    plt.legend()
    plt.xlabel("Structure group")
    plt.ylabel("Total energy (meV/atom, offset arb.)")
    plt.title("Energy scatter within each group of structures")
    plt.savefig("results/energy-scatter.pdf", bbox_inches = "tight")

    # choose representative ID for each structure group
    eigen_IDs = [[s.name for s in group if s.name in E_vs_U]
                 for group in low_e_grouped_structs]
    eigen_IDs = [group[0] for group in eigen_IDs if group]
    print("\nEigen-IDs: {}".format(' '.join(eigen_IDs)))

    plt.figure(figsize = (6,6))
    gsID = "EA573"              # ID of experimental ground state
    def E0(U):
        i = E_vs_U[gsID]['U'].index(U)
        return E_vs_U[gsID]['E'][i]

    for k,v in E_vs_U.items():
        if k not in eigen_IDs:
            continue
        xx = v['U']
        yy = np.asarray(v['E']) - np.asarray([E0(U) for U in xx])
        # yy = v['E']
        plt.plot(xx, yy, 'o-', color = "blue", alpha = 0.5)

    plt.xlim(-0.3, 3.3)
    plt.xlabel("$U - J$ (eV)")
    plt.ylabel("$E - E_0$ (meV/atom)")
    plt.savefig("results/E-vs-U-eigenstructs.pdf", bbox_inches = "tight")

if __name__ == '__main__':
    analyze_trnn2()
    analyze_low_e_structs()
