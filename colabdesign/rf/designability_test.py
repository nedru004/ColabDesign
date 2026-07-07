import os,sys

from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model
from colabdesign.shared.parse_args import parse_args

import pandas as pd
import numpy as np
from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

def get_info(contig):
  F = []
  free_chain = False
  fixed_chain = False
  sub_contigs = [x.split("-") for x in contig.split("/")]
  for n,(a,b) in enumerate(sub_contigs):
    if a[0].isalpha():
      L = int(b)-int(a[1:]) + 1
      F += [1] * L
      fixed_chain = True
    else:
      L = int(b)
      F += [0] * L
      free_chain = True
  return F,[fixed_chain,free_chain]

def parse_contig_string(contigs_str):
  contigs = []
  for contig_str in contigs_str.replace(" ",":").replace(",",":").split(":"):
    if len(contig_str) > 0:
      contig = []
      for x in contig_str.split("/"):
        if x != "0": contig.append(x)
      contigs.append("/".join(contig))
  return contigs

def setup_af_context(contigs, *, copies, rm_aa, initial_guess, use_multimer, af_params_dir):
  chains = alphabet_list[:len(contigs)]
  info = [get_info(x) for x in contigs]
  fixed_pos = []
  fixed_chains = []
  free_chains = []
  both_chains = []
  for pos,(fixed_chain,free_chain) in info:
    fixed_pos += pos
    fixed_chains += [fixed_chain and not free_chain]
    free_chains += [free_chain and not fixed_chain]
    both_chains += [fixed_chain and free_chain]

  flags = {"initial_guess":initial_guess,
           "best_metric":"rmsd",
           "use_multimer":use_multimer,
           "model_names":["model_1_multimer_v3" if use_multimer else "model_1_ptm"],
           "data_dir": af_params_dir}

  if sum(both_chains) == 0 and sum(fixed_chains) > 0 and sum(free_chains) > 0:
    protocol = "binder"
    print("protocol=binder")
    target_chains = []
    binder_chains = []
    for n,x in enumerate(fixed_chains):
      if x: target_chains.append(chains[n])
      else: binder_chains.append(chains[n])
    af_model = mk_af_model(protocol="binder",**flags)
    prep_flags = {"target_chain":",".join(target_chains),
                  "binder_chain":",".join(binder_chains),
                  "rm_aa":rm_aa}
  elif sum(fixed_pos) > 0:
    protocol = "partial"
    print("protocol=partial")
    af_model = mk_af_model(protocol="fixbb",
                           use_templates=True,
                           **flags)
    rm_template = np.array(fixed_pos) == 0
    prep_flags = {"chain":",".join(chains),
                  "rm_template":rm_template,
                  "rm_template_seq":rm_template,
                  "copies":copies,
                  "homooligomer":copies>1,
                  "rm_aa":rm_aa}
  else:
    protocol = "fixbb"
    print("protocol=fixbb")
    af_model = mk_af_model(protocol="fixbb",**flags)
    prep_flags = {"chain":",".join(chains),
                  "copies":copies,
                  "homooligomer":copies>1,
                  "rm_aa":rm_aa}

  if protocol == "binder":
    af_terms = ["plddt","i_ptm","i_pae","rmsd"]
  elif copies > 1:
    af_terms = ["plddt","ptm","i_ptm","pae","i_pae","rmsd"]
  else:
    af_terms = ["plddt","ptm","pae","rmsd"]

  return {"af_model":af_model,
          "protocol":protocol,
          "prep_flags":prep_flags,
          "af_terms":af_terms,
          "fixed_pos":fixed_pos}

def run_designability_test(
    loc,
    pdbs,
    contigs,
    *,
    copies=1,
    num_seqs=8,
    initial_guess=False,
    use_multimer=False,
    use_soluble=False,
    use_antibody=False,
    use_hyper=False,
    num_recycles=3,
    rm_aa="C",
    mpnn_sampling_temp=0.1,
    af_params_dir=".",
):
  """Run MPNN sequence design and AlphaFold2 validation for one or more PDBs.

  Args:
    loc: Output directory for results.
    pdbs: List of input PDB paths, one per design.
    contigs: List of contig strings, one per design (same length as pdbs).
    copies: Number of repeating copies for fixbb protocol.
    num_seqs: Number of MPNN sequences to evaluate per design.
    initial_guess: Initialize AF2 from input coordinates.
    use_multimer: Use AlphaFold multimer model.
    use_soluble: Use soluble MPNN weights.
    use_antibody: Use antibody MPNN weights.
    use_hyper: Use hyper MPNN weights.
    num_recycles: Number of AF2 recycles.
    rm_aa: Amino acids to exclude from MPNN sampling (pass "" or None to allow all).
    mpnn_sampling_temp: MPNN sampling temperature.
    af_params_dir: Directory containing AlphaFold parameters.

  Returns:
    pandas.DataFrame with per-sequence results. Also writes outputs under `loc`.
  """
  if len(pdbs) == 0:
    raise ValueError("pdbs must contain at least one design")
  if len(pdbs) != len(contigs):
    raise ValueError(f"pdbs and contigs must have the same length ({len(pdbs)} vs {len(contigs)})")

  pdbs = [str(p).strip() for p in pdbs]
  contigs = [str(c).strip() for c in contigs]
  if any(len(p) == 0 for p in pdbs) or any(len(c) == 0 for c in contigs):
    raise ValueError("pdbs and contigs must be non-empty strings")

  if rm_aa == "":
    rm_aa = None

  os.makedirs(loc, exist_ok=True)
  os.makedirs(f"{loc}/all_pdb", exist_ok=True)

  batch_size = min(8, num_seqs)

  print("running proteinMPNN...")
  if use_soluble:
    weights = 'soluble'
  elif use_antibody:
    weights = 'antibody'
  elif use_hyper:
    weights = 'hyper'
  else:
    weights = 'original'
  print("Weights Used: ", weights)
  mpnn_model = mk_mpnn_model(weights=weights)

  outs = []
  contexts = []
  for m,(pdb_filename,contig_str) in enumerate(zip(pdbs, contigs)):
    print(f"design {m}: {pdb_filename}")
    parsed_contigs = parse_contig_string(contig_str)
    context = setup_af_context(
        parsed_contigs,
        copies=copies,
        rm_aa=rm_aa,
        initial_guess=initial_guess,
        use_multimer=use_multimer,
        af_params_dir=af_params_dir,
    )
    af_model = context["af_model"]
    protocol = context["protocol"]
    prep_flags = context["prep_flags"]
    fixed_pos = context["fixed_pos"]

    contexts.append(context)
    af_model.prep_inputs(pdb_filename, **prep_flags)
    if protocol == "partial":
      p = np.where(fixed_pos)[0]
      af_model.opt["fix_pos"] = p[p < af_model._len]

    mpnn_model.get_af_inputs(af_model)
    outs.append(mpnn_model.sample(
        num=num_seqs//batch_size,
        batch=batch_size,
        temperature=mpnn_sampling_temp,
    ))

  data = []
  best = {"rmsd":np.inf,"design":0,"n":0}
  print("running AlphaFold...")
  with open(f"{loc}/design.fasta","w") as fasta:
    for m,(out,pdb_filename,contig_str,context) in enumerate(zip(outs,pdbs,contigs,contexts)):
      af_model = context["af_model"]
      prep_flags = context["prep_flags"]
      protocol = context["protocol"]
      fixed_pos = context["fixed_pos"]
      af_terms = context["af_terms"]

      af_model.prep_inputs(pdb_filename, **prep_flags)
      if protocol == "partial":
        p = np.where(fixed_pos)[0]
        af_model.opt["fix_pos"] = p[p < af_model._len]
      for k in af_terms: out[k] = []
      for n in range(num_seqs):
        sub_seq = out["seq"][n].replace("/","")[-af_model._len:]
        af_model.predict(seq=sub_seq, num_recycles=num_recycles, verbose=False)
        for t in af_terms: out[t].append(af_model.aux["log"][t])
        if "i_pae" in out:
          out["i_pae"][-1] = out["i_pae"][-1] * 31
        if "pae" in out:
          out["pae"][-1] = out["pae"][-1] * 31
        rmsd = out["rmsd"][-1]
        if rmsd < best["rmsd"]:
          best = {"design":m,"n":n,"rmsd":rmsd}
        af_model.save_current_pdb(f"{loc}/all_pdb/design{m}_n{n}.pdb")
        af_model._save_results(save_best=True, verbose=False)
        af_model._k += 1
        score_line = [f'design:{m} n:{n}',f'mpnn:{out["score"][n]:.3f}']
        for t in af_terms:
          score_line.append(f'{t}:{out[t][n]:.3f}')
        print(" ".join(score_line)+" "+out["seq"][n])
        line = f'>{"|".join(score_line)}\n{out["seq"][n]}'
        fasta.write(line+"\n")
        row = {"design":m, "n":n, "pdb":pdb_filename, "contig":contig_str,
               "mpnn":out["score"][n], "seq":out["seq"][n]}
        for t in af_terms:
          row[t] = out[t][n]
        data.append(row)
      af_model.save_pdb(f"{loc}/best_design{m}.pdb")

  with open(f"{loc}/best.pdb", "w") as handle:
    remark_text = f"design {best['design']} N {best['n']} RMSD {best['rmsd']:.3f}"
    handle.write(f"REMARK 001 {remark_text}\n")
    handle.write(open(f"{loc}/best_design{best['design']}.pdb", "r").read())

  df = pd.DataFrame(data)
  df.to_csv(f'{loc}/mpnn_results.csv')
  return df

def main(argv):
  ag = parse_args()
  ag.txt("-------------------------------------------------------------------------------------")
  ag.txt("Designability Test")
  ag.txt("-------------------------------------------------------------------------------------")
  ag.txt("REQUIRED")
  ag.txt("-------------------------------------------------------------------------------------")
  ag.add(["pdb="          ],  None,   str, ["input pdb"])
  ag.add(["loc="          ],  None,   str, ["location to save results"])
  ag.add(["contigs="      ],  None,   str, ["contig definition"])
  ag.txt("-------------------------------------------------------------------------------------")
  ag.txt("OPTIONAL")
  ag.txt("-------------------------------------------------------------------------------------")
  ag.add(["copies="       ],         1,    int, ["number of repeating copies"])
  ag.add(["num_seqs="     ],         8,    int, ["number of mpnn designs to evaluate"])
  ag.add(["initial_guess" ],     False,   None, ["initialize previous coordinates"])
  ag.add(["use_multimer"  ],     False,   None, ["use alphafold_multimer_v3"])
  ag.add(["use_soluble"   ],     False,   None, ["use solubleMPNN"])
  ag.add(["use_antibody"  ],     False,   None, ['use antibody MPNN from Dreyer Group'])
  ag.add(["use_hyper"     ],     False,   None, ["use hyperMPNN from Meiler Lab"])
  ag.add(["num_recycles=" ],         3,    int, ["number of recycles"])
  ag.add(["rm_aa="],               "C",    str, ["disable specific amino acids from being sampled"])
  ag.add(["num_designs="  ],         1,    int, ["number of designs to evaluate"])
  ag.add(["mpnn_sampling_temp=" ], 0.1,  float, ["sampling temperature used by proteinMPNN"])
  ag.add(["af_params_dir=" ],      ".",    str, ["directory containing alphafold params"])
  ag.txt("-------------------------------------------------------------------------------------")
  o = ag.parse(argv)

  if None in [o.pdb, o.loc, o.contigs]:
    ag.usage("Missing Required Arguments")

  pdbs, contigs = [], []
  for m in range(o.num_designs):
    if o.num_designs == 1:
      pdb_filename = o.pdb
    else:
      pdb_filename = o.pdb.replace("_0.pdb", f"_{m}.pdb")
    pdbs.append(pdb_filename)
    contigs.append(o.contigs)

  run_designability_test(
      loc=o.loc,
      pdbs=pdbs,
      contigs=contigs,
      copies=o.copies,
      num_seqs=o.num_seqs,
      initial_guess=o.initial_guess,
      use_multimer=o.use_multimer,
      use_soluble=o.use_soluble,
      use_antibody=o.use_antibody,
      use_hyper=o.use_hyper,
      num_recycles=o.num_recycles,
      rm_aa=o.rm_aa,
      mpnn_sampling_temp=o.mpnn_sampling_temp,
      af_params_dir=o.af_params_dir,
  )

if __name__ == "__main__":
   main(sys.argv[1:])
