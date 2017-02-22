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
    # compute (1/2) sum_spin tr{n-n^2} for both atoms
    trnn2 = []
    for dm in DMs:
        if dm['U'] == 0:
            continue
        nn2 = 0
        for Co in [2,3]:
            dm1 = dm['DATA'][-1]['DATA'][Co]['DM1']
            dm2 = dm['DATA'][-1]['DATA'][Co]['DM2']
            nn2 += np.trace(dm1 - np.dot(dm1, dm1)) + \
                   np.trace(dm2 - np.dot(dm2, dm2))
        trnn2.append((dm['NAME'], dm['U'], 0.5 * nn2))
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
    NUM_ATOMS = 8
    plt.plot([1.5,2.0], [1.5,2.0], alpha = 0.5, color = 'black')
    plt.plot([trnn2u1[n] for n in names], [NUM_ATOMS*dE[n]/MEV_IN_EV for n in names],
             'o', color = 'blue', alpha = 0.5)
    plt.xlabel('$(1/2)\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
    plt.ylabel('dE/dU')
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
    plt.xlabel('$(1/2)\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
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
        # yy = np.asarray(v['E']) - np.asarray([E0(U) for U in xx])
        yy = v['E']
        plt.plot(xx, yy, 'o-', color = "blue", alpha = 0.5)

    plt.xlim(-0.3, 3.3)
    plt.xlabel("$U - J$ (eV)")
    plt.ylabel("$E$ (meV/atom)")
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
        # yy = [sum(np.exp(-r/4)/(r-2) for dists in Co_Co_dists(struct) for r in dists) # Ran's heuristic
        #       for struct in structs if struct.name in IDs]
        plt.plot(xx, yy, 'o', alpha = 0.5)
        plt.xlabel('$dE/dU$')
        plt.ylabel('$\sum_{r_i<4} e^{-r_i}/r_i$')
        # plt.ylabel('$\sum_{r_i<4} e^{-r_i/4}/(r_i-2)$')
        plt.title('Heuristic Coulomb vs. Energy Slope Between U = {} and {}eV'.format(U1, U2))
        # plt.show()
        plt.savefig('results/heuristic-Coulomb-simple-{}-{}.pdf'.format(U1, U2), bbox_inches = 'tight')

def analyze_local_geometry():
    # Compute Ran's heuristic, and variations
    with open('data/Co-Co-dists.dat', 'r') as f:
        lines = [line.split() for line in f.readlines()]
    lines = [[line[0], [float(x) for x in line[1:]]] for line in lines]
    dists = pd.DataFrame(lines, columns = "ID distances".split())
    motifs = pd.read_csv("motifs.csv")
    data = pd.merge(dists, motifs, how = 'inner', on = ['ID'])

    energies = pd.read_csv("data/E-vs-U-all-USPEX-Ran.dat")
    Espin = energies[energies.is_spin == True]
    EU0 = Espin[Espin.U_minus_J == 0]
    EU1 = Espin[Espin.U_minus_J == 1]

    sorted_IDs = data.sort_values(by = 'distances').ID
    ax1 = plt.subplot('211')
    ax2 = plt.subplot('212')
    NUM_ATOMS_PER_UC = 8
    MEV_IN_EV = 1000
    for i,ID in enumerate(sorted_IDs):
        dists = data[data.ID == ID].distances.values[0]
        metric = sum([1/r for r in dists if r < 3])
        # heur = sum(np.exp(-r)/r for r in dists)
        ax2.plot([i], metric, 'o', alpha = 0.5)
        # ax2.plot([i]*len(dists), dists, 'o', alpha = 0.5)
        E0 = EU0[EU0.SYSTEM == ID].final_energy.values[0]
        E1 = EU1[EU1.SYSTEM == ID].final_energy.values[0]
        ax1.plot([i], [(E1-E0) * MEV_IN_EV / NUM_ATOMS_PER_UC], 'o', alpha = 0.5)
    plt.xticks(range(len(sorted_IDs)), [ID for ID in sorted_IDs], rotation = 'vertical')
    plt.show()


def analyze_pca():
    # load density matrices
    DMs = ios.read_density_matrices()
    trnn2 = compute_trnn2(DMs)
    trnn2 = pd.DataFrame(trnn2, columns = "SYSTEM U_minus_J trnn2".split())

    # load energies
    energies = pd.read_csv("data/E-vs-U-all-USPEX-Ran.dat")

    # merge energies and tr{n-n^2}
    data = pd.merge(trnn2, energies, how = 'inner', on = ['SYSTEM', 'U_minus_J'])

    # load distances
    with open('data/Co-Co-dists.dat', 'r') as f:
        lines = [line.split() for line in f.readlines()]
    lines = [[line[0], [float(x) for x in line[1:]]] for line in lines]
    dists = pd.DataFrame(lines, columns = "SYSTEM distances".split())


def analyze_compressibility():
    # Compute Ran's heuristic, and variations
    with open('data/Co-Co-dists.dat', 'r') as f:
        lines = [line.split() for line in f.readlines()]
    lines = [[line[0], [float(x) for x in line[1:]]] for line in lines]
    dists = pd.DataFrame(lines, columns = "SYSTEM distances".split()).set_index('SYSTEM')
    dists['metric'] = dists.apply(lambda row: sum([np.exp(-r) for r in row.distances]), axis = 1)
    print(dists.head())

    energies = pd.read_csv("data/E-vs-U-all-USPEX-Ran.dat")
    Espin = energies[energies.is_spin == True]
    EU0 = Espin[Espin.U_minus_J == 0][['SYSTEM', 'final_energy']].set_index('SYSTEM')
    EU1 = Espin[Espin.U_minus_J == 1][['SYSTEM', 'final_energy']].set_index('SYSTEM')
    EU2 = Espin[Espin.U_minus_J == 2][['SYSTEM', 'final_energy']].set_index('SYSTEM')
    EU3 = Espin[Espin.U_minus_J == 3][['SYSTEM', 'final_energy']].set_index('SYSTEM')
    NUM_ATOMS_PER_UC = 8
    for df in [EU0, EU1, EU2, EU3]:
        df.final_energy /= NUM_ATOMS_PER_UC
    EE1 = EU0.join(EU1, lsuffix = '0', rsuffix = '1')
    EE2 = EU2.join(EU3, lsuffix = '2', rsuffix = '3')
    EE = EE1.join(EE2)
    print(EE.head())

    data = EE.join(dists.metric)
    print(data.head())

    from scipy import optimize
    U = [0, 1, 2, 3]
    columns = "a0 a1 a2 s0 s1 s2".split()
    def f(x, a0, a1, a2):
        return a0 + a1*x + a2*x**2
    def fit_parabola(row):
        data = [row.final_energy0, row.final_energy1, row.final_energy2, row.final_energy3]
        popt, pcov = optimize.curve_fit(f, U, data)
        perr = np.sqrt(np.diag(pcov))
        ret = dict(zip(columns, np.hstack((popt, perr))))
        return pd.Series(ret)
    data[columns] = data.apply(fit_parabola, axis = 1)
    print(data)

    ys = [data.a0, data.a1, data.a2]
    yerrs = [data.s0, data.s1, data.s2]
    ylabels = ['$E(U=0)$ (eV)', '$dE/dU$', '$(1/2) d^2E/dU^2$ (1/eV)']
    for i,(y,yerr,ylabel) in enumerate(zip(ys,yerrs,ylabels)):
        plt.figure()
        plt.errorbar(data.metric, y, yerr = yerr, fmt = 'o', alpha = 0.7)
        plt.xlabel('$\sum_i e^{-r_i}$')
        plt.ylabel(ylabel)
        plt.savefig('results/metric-vs-fit{}.pdf'.format(i), bbox_inches = 'tight')


if __name__ == '__main__':
    # analyze_trnn2()
    # analyze_low_e_structs()
    # analyze_by_motif()
    # analyze_heuristic_coulomb()
    # analyze_local_geometry()
    # analyze_pca()
    analyze_compressibility()
