"""Fermi surface from a VASP calculation via IFermi.

Reads band energies on a uniform k-mesh from a vasprun.xml (a static run with a dense,
non-symmetry-reduced Monkhorst-Pack KPOINTS, i.e. ISYM=0 or a full grid), Fourier-
interpolates them, extracts the Fermi surface, and writes an interactive 3D plot.

    python -m fermisurf.vasp_fermi --vasprun vasprun.xml --out fs.html
    python -m fermisurf.vasp_fermi --vasprun vasprun.xml --mu 0.1 --interpolation-factor 8
"""
from __future__ import annotations
import argparse


def build_surface(vasprun="vasprun.xml", interpolation_factor=5, mu=0.0,
                  wigner_seitz=True):
    """Return an IFermi FermiSurface for the given vasprun.xml."""
    from pymatgen.io.vasp.outputs import Vasprun
    from ifermi.interpolate import FourierInterpolator
    from ifermi.surface import FermiSurface
    bs = Vasprun(vasprun).get_band_structure()
    dense_bs = FourierInterpolator(bs).interpolate_bands(interpolation_factor)
    return FermiSurface.from_band_structure(
        dense_bs, mu=mu, wigner_seitz=wigner_seitz)


def main():
    ap = argparse.ArgumentParser(description="VASP Fermi surface (via IFermi)")
    ap.add_argument("--vasprun", default="vasprun.xml",
                    help="vasprun.xml from a uniform-mesh run")
    ap.add_argument("--interpolation-factor", type=float, default=5,
                    help="Fourier interpolation density factor")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="energy offset from E_F in eV (isosurface at E_F + mu)")
    ap.add_argument("--no-wigner-seitz", action="store_true",
                    help="use the reciprocal cell instead of the Wigner-Seitz BZ")
    ap.add_argument("--out", default="fermi_surface.html")
    args = ap.parse_args()

    from ifermi.plot import FermiSurfacePlotter, save_plot
    fs = build_surface(args.vasprun, args.interpolation_factor, args.mu,
                       wigner_seitz=not args.no_wigner_seitz)
    fig = FermiSurfacePlotter(fs).get_plot(plot_type="plotly")
    save_plot(fig, args.out)
    print(f"wrote {args.out} (interp x{args.interpolation_factor}, mu={args.mu} eV)")


if __name__ == "__main__":
    main()
