"""Fermi surface from a SlakoNet (Slater-Koster) tight-binding model.

Samples band energies on a full Monkhorst-Pack grid over the Brillouin zone with a
SlakoNet model, then extracts and plots the Fermi surface (see isosurface.py). Runs on
CPU by default; the grid is dense but each k-point is a cheap eigenproblem.

    python -m fermisurf.slakonet_fermi --poscar POSCAR --nk 24 --out fs.html
    python -m fermisurf.slakonet_fermi --jid JVASP-1002 --nk 24 --model path/to/model
"""
from __future__ import annotations
import argparse
import numpy as np

H2EV = 27.211386245988


def _load_atoms(poscar=None, jid=None):
    from jarvis.core.atoms import Atoms
    if poscar:
        return Atoms.from_poscar(poscar)
    if jid:
        from slakonet.optim import get_atoms
        return get_atoms(jid)[0]
    raise ValueError("Provide --poscar or --jid")


def eigenfield(atoms, nk=24, model_path=None, device="cpu"):
    """Return (field, efermi, cell): band energies (nx,ny,nz,nbands) eV, E_F eV, cell (A)."""
    import torch
    from slakonet.atoms import Geometry
    from slakonet.main import generate_shell_dict_upto_Z65
    if model_path:
        from slakonet.get_bands import load_trained_model
        model = load_trained_model(model_path)
    else:
        from slakonet.optim import default_model
        model = default_model()
    model.eval()

    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    kdims = torch.tensor([[int(nk), int(nk), int(nk)]], dtype=torch.int32)
    with torch.no_grad():
        props, ok = model.compute_multi_element_properties(
            geometry=geometry, shell_dict=shell_dict, kpoints=kdims,
            get_fermi=True, device=device)
    if not ok:
        raise RuntimeError("SlakoNet property computation failed")

    eig = np.real(props["eigenvalues"].detach().cpu().numpy())[0]   # (nk^3, nbands), eV
    occ = np.real(props["occupations"].detach().cpu().numpy())[0]
    nktot, nb = eig.shape
    # Fermi level from band filling (self-consistent with the eigenvalue spectrum;
    # slakonet's reported fermi_energy is on a different reference). Two electrons per
    # state, so fill n_elec/2 states per k-point.
    n_elec = float(occ.sum(axis=1).mean())
    n_filled = int(round(0.5 * n_elec * nktot))
    efermi = float(np.sort(eig.ravel())[min(max(n_filled - 1, 0), eig.size - 1)])
    # SlakoNet MP order (slakonet.atoms._super_sampling): k=(m+0.5)/n, flattened as
    # p = iz*n^2 + iy*n + ix (meshgrid 'ij' over kz,ky,kx). Reshape -> (iz,iy,ix,band),
    # transpose -> (ix,iy,iz,band).
    field = eig.reshape(nk, nk, nk, nb).transpose(2, 1, 0, 3)
    return field, efermi, np.asarray(atoms.lattice_mat, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="SlakoNet tight-binding Fermi surface")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--poscar", help="POSCAR/VASP structure file")
    g.add_argument("--jid", help="JARVIS-DFT id, e.g. JVASP-1002")
    ap.add_argument("--nk", type=int, default=24, help="MP grid density per axis")
    ap.add_argument("--model", default=None, help="SlakoNet model path (default: bundled)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="fermi_surface.html")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    atoms = _load_atoms(args.poscar, args.jid)
    field, efermi, cell = eigenfield(atoms, nk=args.nk, model_path=args.model,
                                     device=args.device)
    from fermisurf.isosurface import fermi_surface_figure, save
    title = args.title or f"{atoms.composition.reduced_formula} Fermi surface (SlakoNet)"
    fig = fermi_surface_figure(field, efermi, cell, title=title)
    save(fig, args.out)
    print(f"E_F = {efermi:.3f} eV | grid {args.nk}^3 | wrote {args.out}")


if __name__ == "__main__":
    main()
