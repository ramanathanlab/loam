#!/usr/bin/env python
"""Small PyRosetta CLI: score / perres / relax / interface.

Run with the pyrosetta_energy venv:
  PY=/flare/FRAME-IDP/msinclair/software/pyrosetta_energy/.venv/bin/python
  $PY rosetta_tool.py score     in.pdb [--json out.json]
  $PY rosetta_tool.py perres    in.pdb --csv perres.csv
  $PY rosetta_tool.py relax     in.pdb --out relaxed.pdb [--rounds 1] [--max-iter 200] [--constrain]
  $PY rosetta_tool.py interface in.pdb --partners A_B [--pack-separated] [--pack-input] [--json out.json]
All energies are ref2015 Rosetta Energy Units (REU). Only numpy is in the venv (no pandas).
"""
import argparse, csv, json, sys, time


def init(extra=""):
    import pyrosetta
    # extra_options APPENDS to the defaults ("-ex1 -ex2aro"); passing a positional string would
    # REPLACE them and change packing/relax/IAM results (verified: BND_0149 dG +86 vs -65).
    pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false " + extra)
    return pyrosetta


def load(pyrosetta, path):
    pose = pyrosetta.pose_from_pdb(path)
    return pose


def cmd_score(a):
    pr = init()
    pose = load(pr, a.pdb)
    sfxn = pr.create_score_function(a.sfxn)
    total = sfxn(pose)
    from pyrosetta.rosetta.core.scoring import ScoreType
    emap = pose.energies().total_energies()
    terms = {}
    for st in sfxn.get_nonzero_weighted_scoretypes():
        name = pr.rosetta.core.scoring.name_from_score_type(st)
        terms[name] = emap[st] * sfxn.get_weight(st)  # weighted
    out = {"pdb": a.pdb, "sfxn": a.sfxn, "nres": pose.total_residue(),
           "chains": pose.num_chains(), "total_score": total, "weighted_terms": terms}
    print(json.dumps(out, indent=2))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


def cmd_perres(a):
    pr = init()
    pose = load(pr, a.pdb)
    sfxn = pr.create_score_function(a.sfxn)
    # Without this, backbone H-bonds are NOT assigned to residues and the per-residue
    # totals do not sum to the pose total (verified: -83.6 vs -129.5 on 6MRR).
    emo = sfxn.energy_method_options()
    emo.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sfxn.set_energy_method_options(emo)
    sfxn(pose)
    sts = list(sfxn.get_nonzero_weighted_scoretypes())
    names = [pr.rosetta.core.scoring.name_from_score_type(s) for s in sts]
    info = pose.pdb_info()
    rows = []
    for i in range(1, pose.total_residue() + 1):
        e = pose.energies().residue_total_energies(i)
        row = {"seqpos": i, "chain": info.chain(i), "pdb_resnum": info.number(i),
               "name3": pose.residue(i).name3(),
               "total": pose.energies().residue_total_energy(i)}  # weighted total
        for n, s in zip(names, sts):
            row[n] = e[s]  # UNWEIGHTED per-term
        rows.append(row)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    worst = sorted(rows, key=lambda r: -r["total"])[:5]
    print("sum(per-res totals)=%.2f  pose total=%.2f" % (sum(r["total"] for r in rows), sfxn(pose)))
    print("worst residues:", [(r["chain"], r["pdb_resnum"], r["name3"], round(r["total"], 2)) for r in worst])


def cmd_relax(a):
    pr = init()
    from pyrosetta.rosetta.protocols.relax import FastRelax
    pose = load(pr, a.pdb)
    sfxn = pr.create_score_function(a.sfxn)
    before = sfxn(pose)
    fr = FastRelax(sfxn, a.rounds)  # standard_repeats = rounds (default protocol uses 5)
    if a.max_iter:
        fr.max_iter(a.max_iter)
    if a.constrain:
        fr.constrain_relax_to_start_coords(True)
    t = time.time()
    fr.apply(pose)
    dt = time.time() - t
    after = sfxn(pose)
    ref = load(pr, a.pdb)
    rmsd = pr.rosetta.core.scoring.CA_rmsd(ref, pose)
    pose.dump_pdb(a.out)
    out = {"pdb": a.pdb, "out": a.out, "score_before": before, "score_after": after,
           "CA_rmsd": rmsd, "relax_seconds": round(dt, 1), "rounds": a.rounds}
    print(json.dumps(out, indent=2))


def cmd_interface(a):
    pr = init()
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    pose = load(pr, a.pdb)
    sfxn = pr.create_score_function(a.sfxn)
    total = sfxn(pose)
    from pyrosetta.rosetta.core.pose import DockingPartners
    # NOTE: this PyRosetta build has NO InterfaceAnalyzerMover(str) constructor;
    # wrap the "A_B" / "AB_C" string in DockingPartners.
    partners = DockingPartners.docking_partners_from_string(a.partners)
    iam = InterfaceAnalyzerMover(partners)
    iam.set_scorefunction(sfxn)
    iam.set_pack_input(a.pack_input)
    iam.set_pack_separated(a.pack_separated)
    iam.set_compute_packstat(True)
    iam.set_calc_dSASA(True)
    iam.set_compute_interface_sc(True)
    iam.apply(pose)
    d = iam.get_all_data()
    out = {"pdb": a.pdb, "partners": a.partners, "total_score": total,
           "interface_dG": iam.get_interface_dG(),  # == d.dG[1]
           "dSASA": iam.get_interface_delta_sasa(),
           "dG_per_dSASAx100": d.dG_dSASA_ratio * 100,
           "shape_complementarity": d.sc_value, "packstat": d.packstat,
           "interface_hbonds": d.interface_hbonds, "delta_unsat_hbonds": d.delta_unsat_hbonds,
           "nres_interface": d.interface_nres[1]}
    print(json.dumps(out, indent=2))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "perres", "relax", "interface"):
        p = sp.add_parser(name)
        p.add_argument("pdb")
        p.add_argument("--sfxn", default="ref2015")
    sp.choices["score"].add_argument("--json")
    sp.choices["perres"].add_argument("--csv", required=True)
    r = sp.choices["relax"]
    r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--max-iter", type=int, default=0)
    r.add_argument("--constrain", action="store_true", help="coordinate-constrain to start (backbone)")
    i = sp.choices["interface"]
    i.add_argument("--partners", default="A_B")
    i.add_argument("--pack-separated", action="store_true", help="repack each partner after separation")
    i.add_argument("--pack-input", action="store_true", help="repack the complex before analysis (modifies pose)")
    i.add_argument("--json")
    a = ap.parse_args()
    t = time.time()
    {"score": cmd_score, "perres": cmd_perres, "relax": cmd_relax, "interface": cmd_interface}[a.cmd](a)
    print("wall_seconds=%.1f" % (time.time() - t), file=sys.stderr)


if __name__ == "__main__":
    main()
