#!/usr/bin/env python
"""MolPatch local (freesasa/MSMS/marching_cubes/Bio.PDB)
========================================================
4-level surface fallback, auto-detect best available.

WSL:  pip install freesasa
Win:  conda install scikit-image

Usage:
  python molpatch_local.py -i protein.pdb -o output/
  python molpatch_local.py -i protein.pdb --method freesasa
"""

import argparse, os, sys, warnings
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.SeqUtils import seq1
from scipy.spatial import KDTree
import networkx as nx

SURFACE_METHODS = {}

# ── 1. freesasa ──
def _surface_freesasa(pdb_file, model):
    import freesasa
    structure = freesasa.Structure(str(pdb_file))
    result = freesasa.calc(structure)
    points = []
    n = result.nAtoms()
    for i in range(n):
        area = result.atomArea(i)
        if area > 0.5:
            reps = max(1, int(area / 2))
            coord = np.array(structure.coord(i))
            points.extend([coord] * reps)
    pts = np.array(points) if points else np.zeros((1, 3))
    print(f"  freesasa: {len(pts)} weighted points from {n} atoms")
    return pts

try:
    import freesasa
    SURFACE_METHODS["freesasa"] = _surface_freesasa
except ImportError:
    pass

# ── 2. MSMS ──
def _surface_msms(pdb_file, model):
    from Bio.PDB.ResidueDepth import get_surface
    surf = get_surface(model, MSMS="msms -density 1.5")
    print(f"  MSMS: {len(surf)} points")
    return np.array(surf)

# MSMS: 需要 Bio.PDB 且 msms 二进制在 PATH
import shutil
if shutil.which("msms") or shutil.which("msms.exe"):
    try:
        from Bio.PDB.ResidueDepth import get_surface
        SURFACE_METHODS["msms"] = _surface_msms
    except Exception:
        pass

# ── 3. marching_cubes (SES via distance field) ──
def _surface_mc(pdb_file, model):
    from skimage.measure import marching_cubes
    # 原子 VDW 半径 (Å)
    rad_map = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
               "H": 1.09, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
    coords, radii = [], []
    for res in Selection.unfold_entities(model, "R"):
        for a in res:
            r = rad_map.get(a.element.upper(), 1.70)
            coords.append(a.get_vector().get_array())
            radii.append(r)
    coords, radii = np.array(coords), np.array(radii)
    probe = 1.4  # 水探针

    # 网格: 0.35 Å 分辨率
    sp = 0.35
    pad = 6.0
    lo, hi = coords.min(0) - pad, coords.max(0) + pad
    shape = tuple(int((hi[i] - lo[i]) / sp) + 1 for i in range(3))
    # 距离场: 每个格点到最近原子球面的有符号距离 (正值=外部)
    # 原子球面 = VDW + probe
    grid = np.full(shape, 1e4, dtype=np.float32)
    expanded = radii + probe
    for x, r in zip(coords, expanded):
        i0 = np.clip(((x - r - sp - lo) / sp).astype(int), 0, np.array(shape) - 1)
        i1 = np.clip(((x + r + sp - lo) / sp).astype(int) + 1, 0, np.array(shape))
        for ix in range(i0[0], i1[0]):
            dx = ix * sp + lo[0] - x[0]
            for iy in range(i0[1], i1[1]):
                dy = iy * sp + lo[1] - x[1]
                for iz in range(i0[2], i1[2]):
                    dz = iz * sp + lo[2] - x[2]
                    d = np.sqrt(dx*dx + dy*dy + dz*dz) - r
                    if d < grid[ix, iy, iz]:
                        grid[ix, iy, iz] = d
    # 等值面 level=0 → SES (球面接触水探针的位置)
    verts, _, _, _ = marching_cubes(grid, level=0.0, spacing=(sp,) * 3)
    verts += lo
    print(f"  marching_cubes SES: {len(verts)} pts (grid {shape}, sp={sp}A)")
    return verts

try:
    from skimage.measure import marching_cubes
    SURFACE_METHODS["marching_cubes"] = _surface_mc
except ImportError:
    pass

# ── 4. Bio.PDB SASA fallback ──
def _surface_sasa(pdb_file, model):
    from Bio.PDB.SASA import ShrakeRupley
    sr = ShrakeRupley()
    sr.compute(model.get_parent(), level="A")
    points = []
    for res in Selection.unfold_entities(model, "R"):
        for a in res:
            if a.sasa > 0.5:
                reps = max(1, int(a.sasa / 5))
                points.extend([a.get_vector().get_array()] * reps)
    pts = np.array(points) if points else np.zeros((1, 3))
    print(f"  Bio.PDB SASA: {len(pts)} points")
    return pts

SURFACE_METHODS["sasa"] = _surface_sasa

def compute_surface(pdb_file, model, method="auto"):
    if method != "auto":
        return SURFACE_METHODS[method](pdb_file, model)
    for name in ["msms", "marching_cubes", "freesasa", "sasa"]:
        if name in SURFACE_METHODS:
            print(f"  Using: {name}")
            return SURFACE_METHODS[name](pdb_file, model)
    raise RuntimeError("No surface method available")

# ── Patch detection ──
class ResiduePatch:
    def __init__(self, residue_ids, dssp_dict, dssp_keys,
                 n_points=0, res_counts=None):
        self.residue_ids = residue_ids
        self.dssp_dict, self.dssp_keys = dssp_dict, dssp_keys
        self.n_points = n_points
        self.res_counts = res_counts or {}
    def size(self): return len(self.residue_ids)
    def get_ids(self): return self.residue_ids
    def frac(self, rid): 
        return self.res_counts.get(rid, 0) / self.n_points if self.n_points else 0
    def count(self, rid):
        return self.res_counts.get(rid, 0)

class ProteinPatch:
    def __init__(self, pdb_id, pdb_file, hydrophobic_res, r=1.25, method="auto"):
        parser = PDBParser(QUIET=True)
        self.model = parser.get_structure(pdb_id, pdb_file)[0]
        self.r = r
        self.hydrophobic = set(hydrophobic_res)
        try:
            self.dssp_dict, self.dssp_keys = dssp_dict_from_pdb_file(pdb_file)
        except Exception:
            self.dssp_dict, self.dssp_keys = {}, []

        # 建立 full_id -> (resname, chain) 映射
        self._res_info = {}
        for chain in self.model:
            for res in chain:
                if seq1(res.get_resname()) != 'X':
                    self._res_info[res.get_full_id()] = (res.get_resname(), chain.id)

        self.G = self._build_graph(pdb_file, method)
        self.patches = self._find_patches()
        print(f"  {len(self.patches)} patches")

    def _sidechain_center(self, atoms):
        return np.array([a.get_vector().get_array() for a in atoms]).mean(0)

    def _build_graph(self, pdb_file, method):
        print("  Computing surface...")
        surf = compute_surface(pdb_file, self.model, method)
        residues = [r for r in Selection.unfold_entities(self.model, "R")
                    if seq1(r.get_resname()) != 'X']
        centers = [self._sidechain_center(r.get_atoms()) for r in residues]
        closest = KDTree(centers).query(surf, k=1)[1]
        G = nx.Graph()
        for i, coord in enumerate(surf):
            aa = seq1(residues[closest[i]].get_resname())
            G.add_node(i, selected=int(aa in self.hydrophobic),
                       coord=coord, res_id=residues[closest[i]].get_full_id(), aa=aa)
        return G

    def _find_patches(self):
        nodes = [i for i in self.G.nodes if self.G.nodes[i]['selected']]
        if not nodes: return []
        coords = [self.G.nodes[i]['coord'] for i in nodes]
        self.G.add_edges_from([(nodes[a], nodes[b]) for a,b in KDTree(coords).query_pairs(self.r)])
        patches = []
        for comp in nx.connected_components(self.G):
            if len(comp) <= 1: continue
            # 统计每个残基在 patch 中的表面点数
            res_counts = {}
            for i in comp:
                rid = self.G.nodes[i]['res_id']
                res_counts[rid] = res_counts.get(rid, 0) + 1
            res_ids = list(res_counts.keys())
            patches.append(ResiduePatch(res_ids, self.dssp_dict, self.dssp_keys,
                                        n_points=len(comp), res_counts=res_counts))
        return sorted(patches, key=lambda x: x.n_points, reverse=True)

# ── CLI ──
def process_pdb(pdb_file, output_dir, residues, r, method):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    out = os.path.join(output_dir, f"patches_{pdb_id}.csv")
    pp = ProteinPatch(pdb_id, pdb_file, residues, r=r, method=method)
    if not pp.patches: return
    rows = []
    aa_3to1 = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G",
               "HIS":"H","ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N",
               "PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V",
               "TRP":"W","TYR":"Y"}
    for i, p in enumerate(pp.patches):
        for rid in p.get_ids():
            resnum = rid[3][1] if len(rid) > 3 else "?"
            icode = (rid[3][2] or "").strip() if len(rid) > 3 and len(rid[3]) > 2 else ""
            residue_id = f"{resnum}{icode}" if icode else str(resnum)
            resname = pp._res_info.get(rid, ("?","?"))[0]
            rows.append({
                "patch_rank": i, "protein_id": pdb_id,
                "residue_id": residue_id,
                "chain": rid[2] if len(rid) > 2 else "?",
                "residue_type": resname,
                "residue_1letter": aa_3to1.get(resname, "?"),
                "patch_n_points": p.n_points,
                "residue_n_points": p.count(rid),
                "frac_of_patch": round(p.frac(rid), 4),
            })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Output: {out} ({len(rows)} rows)")

def main():
    p = argparse.ArgumentParser(description="MolPatch local")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default="molpatch_output")
    p.add_argument("--method", default="auto",
                   choices=["auto","freesasa","msms","marching_cubes","sasa"])
    p.add_argument("--residues", default="ACFILMVWY")
    p.add_argument("--r", type=float, default=1.25)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    residues = list(args.residues.upper())
    # 原子中心法需要更大半径：MSMS 用表面点 (r=1.25)，freesasa/sasa 用原子坐标
    r = args.r
    print(f"Hydrophobic: {residues}  r={r}nm  method={args.method}")
    pdbs = ([os.path.join(args.input, f) for f in os.listdir(args.input)
             if f.endswith(".pdb")] if os.path.isdir(args.input) else [args.input])
    for pdb in pdbs:
        print(f"\nProcessing: {pdb}")
        try:
            process_pdb(pdb, args.output, residues, r, args.method)
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()
