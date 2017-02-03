#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import ios


def main():
    DMs = ios.read_density_matrices()

    # compute sum_spin tr{n-n^2}
    tr_rho = []
    for dm in DMs:
        if dm['U'] > 0:
            dm1 = dm['DATA'][-1]['DATA'][2]['DM1']
            dm2 = dm['DATA'][-1]['DATA'][2]['DM2']
            nn2 = np.trace(dm1 - np.dot(dm1, dm1)) + np.trace(dm2 - np.dot(dm2, dm2))
            tr_rho.append((dm['NAME'], dm['U'], nn2))

    E_vs_U = ios.read_E_vs_U()

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
