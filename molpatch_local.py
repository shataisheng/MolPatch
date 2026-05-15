#!/usr/bin/env python
"""MolPatch 本地版 (无需 Docker)
用法: python molpatch_local.py -i protein.pdb -o output/
"""
import argparse, os
import pandas as pd
from Bio.PDB import PDBParser, Selection
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.SeqUtils import seq1
from scipy.spatial import KDTree
import networkx as nx
import numpy as np


class ResiduePatch:
    def __init__(self, residue_ids, dssp_dict, dssp_keys):
        self.residue_ids = residue_ids
        self.dssp_dict, self.dssp_keys = dssp_dict, dssp_keys
    def size(self): return len(self.residue_ids)
    def get_ids(self): return self.residue_ids
    def residues(self):
        return [self.dssp_dict[k][1] for k in self.dssp_keys
                if k in self.dssp_dict and self.dssp_dict[k] is not None]


class ProteinPatch:
    def __init__(self, pdb_id, pdb_file, hydrophobic_res, r=1.25,
                 msms_cmd="msms -density 1.5"):
        parser = PDBParser(QUIET=True)
        self.model = parser.get_structure(pdb_id, pdb_file)[0]
        self.r = r
        self.msms = msms_cmd if msms_cmd.startswith("msms") else "msms " + msms_cmd
        self.hydrophobic = set(hydrophobic_res)

        try:
            self.dssp_dict, self.dssp_keys = dssp_dict_from_pdb_file(pdb_file)
        except Exception:
            self.dssp_dict, self.dssp_keys = {}, []

        self.G = self._build_graph()
        self.patches = self._find_patches()
        print(f"  {len(self.patches)} patches found")

    def _sidechain_center(self, atoms):
        v = [a.get_vector().get_array() for a in atoms]
        return np.array(v).mean(axis=0)

    def _build_graph(self):
        print("  Computing MSMS surface...")
        surf = get_surface(self.model, MSMS=self.msms)
        print(f"  Surface points: {len(surf)}")

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
        if not nodes:
            return []
        coords = [self.G.nodes[i]['coord'] for i in nodes]
        pairs = KDTree(coords).query_pairs(self.r)
        self.G.add_edges_from([(nodes[a], nodes[b]) for a, b in pairs])

        patches = []
        for comp in nx.connected_components(self.G):
            if len(comp) <= 1:
                continue
            sub = self.G.subgraph(comp)
            res_ids = list(set(sub.nodes[i]['res_id'] for i in sub.nodes))
            patches.append(ResiduePatch(res_ids, self.dssp_dict, self.dssp_keys))
        return sorted(patches, key=lambda x: x.size(), reverse=True)


def process_pdb(pdb_file, output_dir, residues, msms_cmd):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    out = os.path.join(output_dir, f"patches_{pdb_id}.csv")

    pp = ProteinPatch(pdb_id, pdb_file, residues, msms_cmd=msms_cmd)
    if not pp.patches:
        print("  WARNING: no patches found")
        return

    rows = []
    for i, p in enumerate(pp.patches):
        for rid in p.get_ids():
            rows.append({
                "patch_rank": i, "protein_id": pdb_id,
                "residue_id": rid[3][1] if len(rid) > 3 else "?",
                "chain": rid[2] if len(rid) > 2 else "?",
                "residue_type": rid[3][0] if len(rid) > 3 else "?",
                "patch_size": p.size(),
            })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Output: {out} ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description="MolPatch local")
    p.add_argument("-i", "--input", required=True, help="PDB file or directory")
    p.add_argument("-o", "--output", default="molpatch_output")
    p.add_argument("--msms", default="msms -density 1.5", help="MSMS command")
    p.add_argument("--residues", default="ACFILMVWY", help="Hydrophobic residues (1-letter)")
    p.add_argument("--r", type=float, default=1.25, help="Radius (nm)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    residues = list(args.residues.upper())
    print(f"Hydrophobic: {residues}  Radius: {args.r} nm")

    if os.path.isdir(args.input):
        pdbs = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".pdb")]
    else:
        pdbs = [args.input]

    for pdb in pdbs:
        print(f"\nProcessing: {pdb}")
        try:
            process_pdb(pdb, args.output, residues, args.msms)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
