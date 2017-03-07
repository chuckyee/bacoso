#!/usr/bin/env python

from __future__ import division, print_function

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

from pymatgen.io.vasp import Vasprun
import ios


def compute_trnn2(dm, atoms):
    """
    Compute (1/2) sum_spin tr{n-n^2} over specified atoms for final iteration

    Args:
        dm (dict): data structure produced by read_outcar_density_matrix()
        atoms (list): index of atoms to include in sum
    """
    trnn2 = 0
    for atom in atoms:
        dm1 = dm[-1]['DATA'][atom]['DM1']
        dm2 = dm[-1]['DATA'][atom]['DM2']
        trnn2 += np.trace(dm1 - np.dot(dm1, dm1)) + \
                 np.trace(dm2 - np.dot(dm2, dm2))
    return trnn2 / 2


def extract_data(rundir, atoms):
    foutcar = os.path.join(rundir, 'OUTCAR')
    dms = ios.read_outcar_density_matrix(foutcar)
    fvasprun = os.path.join(rundir, 'vasprun.xml')
    v = Vasprun(fvasprun)
    data = {
        'E': v.final_energy,
        'trnn2': compute_trnn2(dms, atoms)
        }
    return data


def process_runs():
    Co_atoms = [2, 3]
    Ustrs = '0.0 0.1 0.2 0.3 0.4 0.5'.split()
    for Ustr in Ustrs:
        rundirs = glob.glob('inputs/U{}/EA*/*chuckyee'.format(Ustr))
        names = []
        data = []
        for rundir in rundirs:
            _, _, name, _ = rundir.split('/')
            names.append(name)
            data.append(extract_data(rundir, Co_atoms))
        keys = data[0].keys()
        df = pd.DataFrame(
            dict(zip(keys, [[x[k] for x in data] for k in keys])),
            index=names, columns=keys)
        df.to_csv('data/{}.csv'.format(Ustr))


def plot_eslope_vs_trnn2():
    U0 = pd.read_csv('data/0.0.csv', index_col = 0)    
    U1 = pd.read_csv('data/0.1.csv', index_col = 0)    
    plt.plot([1.5,1.8], [1.5,1.8], '-', alpha = 0.3, color = 'black')
    xx = U0.trnn2
    yy = (U1.E - U0.E)/0.1
    plt.plot(xx, yy, 'o', alpha = 0.5)
    # for i,txt in enumerate(U0.index):
    #     plt.text(xx[i], yy[i], txt)
    plt.xlim(1.4, 1.9)
    plt.ylim(1.4, 1.9)
    plt.xlabel('$(1/2)\sum_\sigma\mathrm{tr}\{n_\sigma - n_\sigma^2\}$')
    plt.ylabel('dE/dU')
    plt.title('Energy Change vs. Itineracy')
    plt.savefig('results/u0-eslope-vs-trnn2.pdf', bbox_inches = 'tight')


def plot_curvature_vs_trnn2():
    Us = np.linspace(0, 0.5, 6)
    dfs = [pd.read_csv('data/{}.csv'.format(U), index_col=0) for U in Us]

    columns = "a0 a1 a2 s0 s1 s2".split()

    def f(x, a0, a1, a2):
        return a0 + a1*x + a2*x**2

    def fit_parabola(xs, ys):
        popt, pcov = optimize.curve_fit(f, xs, ys)
        perr = np.sqrt(np.diag(pcov))
        return dict(zip(columns, np.hstack((popt, perr))))

    names = dfs[0].index
    fits = []
    for name in names:
        xs = Us
        ys = [df.loc[name].E for df in dfs]
        fit_params = fit_parabola(xs, ys)
        fits.append(fit_params)

    keys = fits[0].keys()
    df = pd.DataFrame(
        dict(zip(keys, [[x[k] for x in fits] for k in keys])),
        index=names, columns=keys)

    # Compute Ran's heuristic
    def metric(rs):
        return sum([np.exp(-r) for r in rs])

    with open('data/Co-Co-dists.dat', 'r') as f:
        lines = [line.split() for line in f.readlines()]
    lines = [[line[0], metric([float(x) for x in line[1:]])] for line in lines]
    metric = pd.DataFrame(lines, columns=["name", "metric"]).set_index('name')
    df = df.join(metric)

    x = df.metric
    ys = [df.a0, df.a1, df.a2]
    yerrs = [df.s0, df.s1, df.s2]
    ylabels = ['$E(U=0)$ (eV)', '$dE/dU$', '$(1/2) d^2E/dU^2$ (1/eV)']
    for i,(y,yerr,ylabel) in enumerate(zip(ys,yerrs,ylabels)):
        plt.figure()
        plt.errorbar(x, y, yerr = yerr, fmt = 'o', alpha = 0.7)
        plt.xlabel('$\sum_i e^{-r_i}$')
        plt.ylabel(ylabel)
        plt.savefig('results/u0-metric-vs-fit{}.pdf'.format(i), bbox_inches = 'tight')


if __name__ == '__main__':
    # process_runs()
    plot_eslope_vs_trnn2()
    # plot_curvature_vs_trnn2()
