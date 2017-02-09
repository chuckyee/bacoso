#!/usr/bin/env python

from __future__ import division, print_function

import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pymatgen import Element
from pymatgen.analysis.structure_matcher import StructureMatcher

import ios

# TODO:
#  - organize data into single Pandas dataframe
#  - cache data on load


def compute_trnn2(DMs):
    # compute sum_spin tr{n-n^2}
    trnn2 = []
    for dm in DMs:
        if dm['U'] > 0:
            dm1 = dm['DATA'][-1]['DATA'][2]['DM1']
            dm2 = dm['DATA'][-1]['DATA'][2]['DM2']
            nn2 = np.trace(dm1 - np.dot(dm1, dm1)) + \
                  np.trace(dm2 - np.dot(dm2, dm2))
            trnn2.append((dm['NAME'], dm['U'], nn2))
    return trnn2


def analyze_trnn2():
    DMs = ios.read_density_matrices()
    E_vs_U = ios.read_E_vs_U()
    trnn2 = compute_trnn2(DMs)

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

    # Scatter plot of E(U=0) vs. tr{n-n^2} -- no correlation visible
    plt.figure()
    trnn2u1 = dict((n, c) for n,u,c in trnn2 if u == 1.0)
    E = dict((n, E_vs_U[n]['E'][0]) for n in trnn2u1)
    names = trnn2u1.keys()
    MEV_IN_EV = 1000
    plt.plot([trnn2u1[n] for n in names], [E[n]/MEV_IN_EV for n in names],
             'o', color = 'blue', alpha = 0.5)
    plt.xlabel('$\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
    plt.ylabel('E(U = 0) (eV/atom)')
    plt.title('Total Energy vs. Itineracy')
    plt.savefig('results/E0-vs-trnn2.pdf', bbox_inches = 'tight')


def analyze_low_e_structs():
    df = ios.read_individuals_file('data/Individuals')
    E_vs_U = ios.read_E_vs_U()
    with open('data/grouped-structs.pickle', 'rb') as f:
        grouped_structs = pickle.load(f)

    # Isolate unique low-energy structures
    energy_cutoff = 0.5         # eV / unit cell
    low_energy_IDs = df[df.Enthalpy < min(df.Enthalpy) + energy_cutoff].ID
    low_energy_IDs = ['EA{}'.format(id) for id in low_energy_IDs]

    low_e_grouped_structs = [[s for s in group if s.name in low_energy_IDs]
                             for group in grouped_structs]
    low_e_grouped_structs = [group for group in low_e_grouped_structs if group]

    if False:
        # Not used: faster to use cached grouped structures read above
        structs = ios.read_consolidated_poscars('data/gatheredPOSCARS')
        low_energy_structs = [s for s in structs if s.name in low_energy_IDs]
        sm = StructureMatcher()
        low_e_grouped_structs = sm.group_structures(low_energy_structs)

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

def analyze_by_motif():
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

    # stem and leaf plot (of sorts)
    for group in low_e_grouped_structs:
        IDs = [struct.name for struct in group if struct.name in E_vs_U]
        if IDs:
            print("{:2} | {}".format(len(IDs), ' '.join(IDs)))


    DMs = ios.read_density_matrices()
    trnn2 = compute_trnn2(DMs)

    df = pd.read_csv("motifs.csv")
    grouped_df = df.groupby("Group")
    grouped_IDs = [(i, [ID for ID in data.ID]) for i,data in grouped_df]

    # Scatter plot of E(1)-E(0) vs. tr{n-n^2}
    plt.figure()
    trnn2u1 = dict((n, c) for n,u,c in trnn2 if u == 1.0)
    dE = dict((n, E_vs_U[n]['E'][1] - E_vs_U[n]['E'][0]) for n in trnn2u1)
    MEV_IN_EV = 1000
    for group,IDs in grouped_IDs:
        plt.plot([trnn2u1[n] for n in IDs], [dE[n]/MEV_IN_EV for n in IDs],
                 'o', alpha = 0.5, label = str(group))
    plt.legend(loc = 'best')
    plt.xlabel('$\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
    plt.ylabel('dE/dU (per atom)')
    plt.title('Energy Change vs. Itineracy')
    plt.savefig('results/motifs-dE-vs-trnn2.pdf', bbox_inches = 'tight')

def analyze_heuristic_coulomb():
    DMs = ios.read_density_matrices()
    trnn2 = compute_trnn2(DMs)
    structs = ios.read_consolidated_poscars('data/gatheredPOSCARS')

    RMAX = 4.0
    Co = Element('Co')
    def min_Co_Co_dist(struct):
        Co_sites = [site for site in struct if site.specie == Co]
        dist = min(
            min(dist for neigh,dist in struct.get_neighbors(site, RMAX)
                if neigh.specie == Co)
            for site in Co_sites)
        return dist

    def Co_Co_dists(struct):
        Co_sites = [site for site in struct if site.specie == Co]
        dists = [
            sorted([dist for neigh,dist in struct.get_neighbors(site, RMAX)
                if neigh.specie == Co])
            for site in Co_sites]
        print('{:5} {}'.format(struct.name,
                               ' '.join('{:10f}'.format(x) for x in dists[0])))
        return dists

    def get_trnn2(ID, U):
        data = [x for n,u,x in trnn2 if n == ID and u == U]
        return data[0] if data else None

    IDs = [name for name,U,x in trnn2 if U == 1.0]
    E_vs_U = ios.read_E_vs_U()
    MEV_IN_EV = 1000
    U1 = 0
    U2 = 1
    for U1,U2 in [(0,1), (1,2), (2,3)]:
        xx = [(E_vs_U[struct.name]['E'][U2] - E_vs_U[struct.name]['E'][U1])/MEV_IN_EV
              for struct in structs if struct.name in IDs]
        yy = [sum(np.exp(-r)/r for dists in Co_Co_dists(struct) for r in dists)
              for struct in structs if struct.name in IDs]
        # yy = [sum(np.exp(-r/4)/(r-2) for dists in Co_Co_dists(struct) for r in dists)
        #       for struct in structs if struct.name in IDs]
        plt.plot(xx, yy, 'o', alpha = 0.5)
        plt.xlabel('$dE/dU$ (eV)')
        plt.ylabel('$\sum_{r_i<4} e^{-r_i}/r_i$')
        # plt.ylabel('$\sum_{r_i<4} e^{-r_i/4}/(r_i-2)$')
        plt.title('Heuristic Coulomb vs. Energy Slope Between U = {} and {}eV'.format(U1, U2))
        # plt.show()
        plt.savefig('results/heuristic-Coulomb-simple-{}-{}.pdf'.format(U1, U2), bbox_inches = 'tight')


def analyze_local_geometry():
    pass

if __name__ == '__main__':
    # analyze_trnn2()
    # analyze_low_e_structs()
    # analyze_by_motif()
    analyze_heuristic_coulomb()
    # analyze_local_geometry()
