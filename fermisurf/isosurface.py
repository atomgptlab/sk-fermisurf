"""Marching-cubes Fermi-surface extraction + interactive 3D plotting.

Given band energies sampled on a regular Monkhorst-Pack grid over the Brillouin
zone, extract the E = E_F isosurface for every band that crosses the Fermi level and
render them as interactive 3D meshes with Plotly. Backend-agnostic: the slakonet and
VASP front-ends both feed it the same (nx, ny, nz, nbands) field.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes

# a colourblind-safe categorical cycle for bands
_BAND_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2",
                "#72B7B2", "#9D755D", "#BAB0AC"]


def reciprocal_matrix(cell):
    """Reciprocal lattice (columns b1,b2,b3, with 2*pi) from a real cell (rows a1,a2,a3)."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T


def _band_mesh(field, efermi, recip, color, name):
    """One band -> a Plotly Mesh3d of its E=efermi isosurface, or None if it doesn't cross."""
    if field.min() > efermi or field.max() < efermi:
        return None
    nx, ny, nz = field.shape
    # wrap one layer so the surface closes across periodic BZ boundaries
    f = np.pad(field, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    try:
        verts, faces, _, _ = marching_cubes(
            f, level=float(efermi), spacing=(1.0 / nx, 1.0 / ny, 1.0 / nz))
    except (ValueError, RuntimeError):
        return None
    frac = verts - 0.5                 # centre the zone on Gamma: [0,1) -> [-0.5,0.5)
    cart = frac @ recip.T              # fractional -> Cartesian k (1/Angstrom)
    return go.Mesh3d(
        x=cart[:, 0], y=cart[:, 1], z=cart[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=1.0, flatshading=True, name=name,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2),
        showscale=False, hoverinfo="name")


def fermi_surface_figure(field4d, efermi, cell, title="Fermi surface",
                         bz_edges=True):
    """Build a Plotly figure of the Fermi surface.

    field4d : (nx, ny, nz, nbands) band energies in eV on a full BZ grid.
    efermi  : Fermi energy in eV.
    cell    : 3x3 real-space lattice (rows, Angstrom).
    """
    field4d = np.asarray(field4d, dtype=float)
    recip = reciprocal_matrix(cell)
    traces = []
    nb = field4d.shape[3]
    for b in range(nb):
        m = _band_mesh(field4d[..., b], efermi, recip,
                       _BAND_COLORS[b % len(_BAND_COLORS)], f"band {b + 1}")
        if m is not None:
            traces.append(m)
    if not traces:
        raise RuntimeError(
            "No band crosses E_F on this grid; check the Fermi energy / density.")
    if bz_edges:
        traces += _bz_wire(recip)
    fig = go.Figure(traces)
    fig.update_layout(
        title=title, showlegend=True,
        scene=dict(xaxis_title="k_x (1/Å)", yaxis_title="k_y (1/Å)",
                   zaxis_title="k_z (1/Å)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _bz_wire(recip):
    """A light wireframe of the reciprocal-cell parallelepiped (not the true WS cell,
    but a cheap orientation guide)."""
    b = recip.T  # rows b1,b2,b3
    corners = np.array([i * b[0] + j * b[1] + k * b[2]
                        for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    xs, ys, zs = [], [], []
    for a, c in edges:
        xs += [corners[a, 0], corners[c, 0], None]
        ys += [corners[a, 1], corners[c, 1], None]
        zs += [corners[a, 2], corners[c, 2], None]
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                         line=dict(color="rgba(80,80,80,0.5)", width=2),
                         name="recip. cell", hoverinfo="skip")]


def save(fig, out_html):
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html
