#!/usr/bin/env python3
"""
Python / matplotlib port of the gnuplot script for the mixing free energy
(Delta F_mix*), optimal phase fraction, funneling fraction and photoemission
peak vs. bromine fraction x.

Usage:
    python fmix_plots.py                 # 1000 samples, show window, save PNG
    python fmix_plots.py --samples 300   # faster, coarser curves
    python fmix_plots.py --no-show       # just save the PNG
"""
import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# ----------------------------------------------------------------------------
# Parameters (same names/values as the gnuplot script)
# ----------------------------------------------------------------------------
ns = [1e-4, 3.16e-5, 1e-5, 3.16e-5, 3.16e-5]
Ts = [0.0258, 0.0258, 0.0258, 0.0258, 0.0275]
Us = [0.125, 0.125, 0.125, 0.110, 0.125]
Efs = [1.75, 1.70, 1.65, 1.70, 1.70]          # overwritten below from n_unit
Colors = ["blue", "green", "red", "cyan", "magenta"]
nplot = len(ns)

# gnuplot's "green" is pure #00ff00 (matplotlib's "green" is darker)
GNUPLOT_COLORS = {"blue": "#0000ff", "green": "#00ff00", "red": "#ff0000",
                  "cyan": "#00ffff", "magenta": "#ff00ff"}

pi = 3.141592654
a = 1.225            # gives a band edge density of states of 1.8e19 at RT
aprime = 0.81374
parent = 0.6
chi = 0.641
RT = 0.0258
iterations = 3
intervals = 100
epsilon = 1e-10
selfconsistent = 1
WARN_CONVERGENCE = False   # gnuplot script printed a line if change > 1e-5

NAN = float("nan")


# ----------------------------------------------------------------------------
# Physics functions
# ----------------------------------------------------------------------------
def Eg(x):
    return 1.57 + 0.33 * x + 0.39 * x * x


def DeltaEg(x, xi):
    return Eg(x) - Eg(xi)


def N(muprime, T):
    if muprime > chi * T:
        return (2.0 / 3.0 * a * muprime ** 1.5
                + pi ** 2 / 12 * T ** 2 * a * muprime ** (-0.5))
    return aprime * a * math.sqrt(pi) / 2 * T ** 1.5 * math.exp(muprime / T)


def FF(x, xi, phi, mu, T):
    num = phi * N(mu - Eg(x), T)
    return num / (num + (1 - phi) * N(mu - Eg(xi), T))


def d(T):
    return pi ** 2 / 4.0 * T ** 2


def EPL(x, muprime, T):
    if muprime > chi * T:
        return (Eg(x) + 3.0 / 5.0 * muprime
                + 12.0 / 5.0 * muprime * d(T) / (2.0 * muprime ** 2 + d(T))
                - 3.0 / 2.0 * T)
    return Eg(x)


def FMixstar(x, xi, phi, U, T, fI, nunit, muprime, DEg):
    if phi > 2 * epsilon:
        return (phi / (1 - phi) * (x - xi) ** 2
                * (-U + 3.0 / 2.0 * T / (xi * (1.0 - xi)))
                + fI * nunit * (EPL(x, muprime, T) - Eg(xi)))
    return 0.0


def Efshift(ff, T):
    one_minus = 1 - ff
    if one_minus <= 0:                 # gnuplot: log(0) = -inf
        return -math.inf
    return T * math.log(one_minus)


def Effromnunit(nunit, xi, T):
    return Eg(xi) + T * math.log(2 * nunit / (T ** 1.5 * a * aprime * math.sqrt(pi)))


# ----------------------------------------------------------------------------
# Self-consistent chemical potential and free energy
# ----------------------------------------------------------------------------
def _selfconsistent(x, xi, phi, T, muinit):
    """Fixed-point iteration. Returns (mu, fillingFactor, change) where the
    filling factor is the one from the *last* iteration (before the final mu
    update), exactly as in the gnuplot function."""
    mu = muinit
    ff = NAN
    change = 0.0
    for _ in range(iterations):
        ff = FF(x, xi, phi, mu, T)
        shift = Efshift(ff, T)
        change = muinit + shift - mu
        mu = mu + change
    if WARN_CONVERGENCE and change > 1e-5:
        print(x, xi, phi, T, muinit, change)
    return mu, ff, change


def FMixstarsc(x, xi, phi, U, T, nunit, muinit, DEg):
    try:
        mu, ff, _ = _selfconsistent(x, xi, phi, T, muinit)
        return FMixstar(x, xi, phi, U, T, ff, nunit, mu - Eg(x), DEg)
    except (ValueError, ZeroDivisionError, OverflowError):
        return NAN


def musc(x, xi, T, muinit, phi):
    try:
        return _selfconsistent(x, xi, phi, T, muinit)[0]
    except (ValueError, ZeroDivisionError, OverflowError):
        return NAN


def _F(x, xi, phi, U, T, nunit, mu):
    """Free energy at a given phi, self-consistent or not (as in the script)."""
    if selfconsistent:
        return FMixstarsc(x, xi, phi, U, T, nunit, mu, DeltaEg(x, xi))
    return FMixstar(x, xi, phi, U, T, FF(x, xi, phi, mu, T), nunit,
                    mu - Eg(x), DeltaEg(x, xi))


# ----------------------------------------------------------------------------
# Root finding: $BinarySearch* / $Lowest_Root_* merged into one routine.
# Returns (Fmix, phi): Fmix = $Lowest_Root_FMix, phi = $Lowest_Root_phi
# (phi is NaN if the optimum is below 3*epsilon, as in $BinarySearchphi).
# ----------------------------------------------------------------------------
def _binary_search(x, xi, U, T, nunit, mu, lo, hi):
    phimin, phimax = lo, hi
    E1 = E2 = NAN
    phiguess = 0.5 * (lo + hi)
    while (phimax - phimin) / (phimax + phimin) > epsilon:
        phiguess = (phimin + phimax) / 2.0
        E1 = _F(x, xi, phiguess - epsilon, U, T, nunit, mu)
        E2 = _F(x, xi, phiguess + epsilon, U, T, nunit, mu)
        if E2 > E1:
            phimax = phiguess
        else:
            phimin = phiguess
    return (E1 + E2) / 2.0, phiguess


def lowest_root(x, xi, U, T, nunit, mu):
    phiguess = epsilon
    ratio = (1.0 / epsilon) ** (1.0 / intervals)
    Elast = _F(x, xi, phiguess, U, T, nunit, mu)
    for _ in range(intervals):
        phiguess *= ratio
        Ecurr = _F(x, xi, phiguess, U, T, nunit, mu)
        if Ecurr > Elast:      # this or the previous interval holds the minimum
            Fmix, phi = _binary_search(x, xi, U, T, nunit, mu,
                                       phiguess / ratio / ratio, phiguess)
            if phi < 3 * epsilon:
                return Fmix, NAN
            return Fmix, phi
        Elast = Ecurr
    return NAN, NAN


# ----------------------------------------------------------------------------
# Curve computation (one worker per curve so the 5 curves run in parallel)
# ----------------------------------------------------------------------------
def compute_curve(args):
    i, xs = args
    n, U, T, Ef = ns[i], Us[i], Ts[i], Efs[i]
    Fmix = np.full(len(xs), NAN)
    phi = np.full(len(xs), NAN)
    ffun = np.full(len(xs), NAN)
    for k, x in enumerate(xs):
        Fm, ph = lowest_root(x, parent, U, T, n, Ef)
        Fmix[k], phi[k] = Fm, ph
        mu = musc(x, parent, T, Ef, ph)
        try:
            ffun[k] = FF(x, parent, ph, mu, T)
        except (ValueError, ZeroDivisionError, OverflowError):
            ffun[k] = NAN
    return i, Fmix / phi, phi, ffun


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", default="fmix_plots.png")
    args = ap.parse_args()

    # Fermi levels from the n_unit values (overrides the initial Efs)
    for i in range(nplot):
        Efs[i] = Effromnunit(ns[i], parent, Ts[i])

    # --- sanity prints from the gnuplot script ---
    print(lowest_root(0.29, parent, Us[0], Ts[0], ns[0], Efs[0])[0])
    print(lowest_root(0.55, parent, Us[0], Ts[0], ns[0], Efs[0])[0])
    for i in range(nplot):
        _, phi = lowest_root(0.0, parent, Us[i], Ts[i], ns[i], Efs[i])
        mu = musc(0.0, parent, Ts[i], Efs[i], phi)
        fI = FF(0.0, parent, phi, mu, Ts[i])
        print(ns[i], Us[i], Ts[i], mu - Eg(0.0), phi, fI,
              phi / (1 - phi) * (parent - 0) ** 2 * -Us[i],
              phi / (1 - phi) * (parent - 0) ** 2 * 3.0 / 2.0 * Ts[i] / parent / (1 - parent),
              fI * ns[i] * (EPL(0.0, mu - Eg(0.0), Ts[i]) - Eg(parent)))

    xs = np.linspace(0.0, parent, args.samples)
    with ProcessPoolExecutor() as ex:
        results = dict((r[0], r[1:]) for r in ex.map(compute_curve,
                                                      [(i, xs) for i in range(nplot)]))

    # ---------------------------------------------------------------- plotting
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 15, "axes.linewidth": 2,
                         "mathtext.default": "regular"})
    fig, axes = plt.subplots(1, 4, figsize=(19.2, 7.2))
    left, right, bottom = 0.10, 0.95, 0.12
    top = 1 - 0.06 * (nplot + 1)
    spacing = 0.08
    w = ((right - left) - 3 * spacing) / 4
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=spacing / w)

    col = [GNUPLOT_COLORS[c] for c in Colors]
    ax1, ax2, ax3, ax4 = axes
    for i in range(nplot):
        ax1.plot(xs, results[i][0], color=col[i], lw=2)
        ax2.plot(xs, results[i][1], color=col[i], lw=2)
        ax3.plot(xs, results[i][2], color=col[i], lw=2)
        ax4.plot(xs, [EPL(x, Efs[i] - Eg(x), Ts[i]) for x in xs], color=col[i], lw=2)
    ax4.plot(xs, Eg(xs), color="black", lw=2, ls=(0, (4, 2.5)))

    ax1.set_ylabel(r"$\Delta F_{mix}^{*}/\phi$ (eV/form. unit)", labelpad=10)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%0.1e"))
    ax2.set_ylabel("Optimal Phase Fraction")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 1e-2)
    ax3.set_ylabel("Funneling Fraction")
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
    ax4.set_ylabel("Photoemission Peak (eV)")
    ax4.yaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
    for ax in axes:
        ax.set_xlim(0, parent)
        ax.tick_params(width=2, direction="in", top=True, right=False)
        ax.tick_params(which="minor", width=1.5)

    # Shared legend (the "dummy plot" in the gnuplot script)
    handles, labels = [], []
    for i in range(nplot):
        T_C = Ts[i] * 298.15 / 0.0258 - 273.15
        labels.append(f"$n_{{unit}}$={ns[i]:.0e}, U={1000 * Us[i]:3.0f}meV, "
                      f"μ={Efs[i]:.2f}eV, T={T_C:3.0f}°C")
        handles.append(Line2D([], [], color=col[i], lw=2))
    labels.append("Emissive Bandgap")
    handles.append(Line2D([], [], color="black", lw=2, ls=(0, (4, 2.5))))
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, top + 0.01))

    fig.text(0.5, 0.02, "Bromine Fraction (x)", ha="center", va="bottom")

    fig.savefig(args.out, dpi=100)
    print("saved", args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
