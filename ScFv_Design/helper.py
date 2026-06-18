from Bio import SeqIO
from antpack import SingleChainAnnotator
import pandas as pd
import os

def get_anti_seqs_from_fasta(fasta_path: str):
    """ Function to return a nested dictionary of initial key being antibody names/IDs and value being dictionary of {"heavy": hc_seq, "light": lc_seq}
    Args:
        fasta_path (str): path to fasta file containing antibody sequences
    Returns:
        dict: nested dictionary of antibody sequences 
    """

    heavy_seqs = {}
    light_seqs = {}
    anti_ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        if "heavy" in record.id:
            heavy_seqs[record.id] = str(record.seq)
        elif "light" in record.id:
            light_seqs[record.id] = str(record.seq)
        seq_id = '_'.join(record.id.split("_")[0:2])
        anti_ids.append(seq_id)
    unique_antibodies = list(set(anti_ids))

    anti_seqs = {}
    for anti_id in unique_antibodies:
        anti_seqs[anti_id] = {"heavy": heavy_seqs[f"{anti_id}_heavy"], "light": light_seqs[f"{anti_id}_light"]}
    
    return anti_seqs

def heatmap_setup(seq, scheme: str, chain: str):
    """ Function to cleanly do setup code for generating heatmaps to store as pngs to accompany each run in MLFlow"""
    antpack_annotator = SingleChainAnnotator(chains = [chain], # Specify chain with options for heavy: H & options for light: [L,K] 
                                             scheme = scheme) # Specify annotation scheme: imgt, aho, martin (modern chothia), or kabat
    # Numbering Antibody Respective Chain Seqs based on scheme provided
    annotated_seqs = antpack_annotator.analyze_seqs(seq)
    seq_numbered_list, percent_identity_list, chain_detect_list, error_list = zip(*annotated_seqs)

    # Build MSA using seqs remaining after filtering (if required):
    pos_codes_list, aligned_seqs_list = antpack_annotator.build_msa(sequences = seq, annotations = annotated_seqs)
    # Assign CDR or FR labels based on pos_codes_list
    scheme_cdr_labels = antpack_annotator.assign_cdr_labels(numbering = pos_codes_list, chain = chain, scheme = scheme)

    # Define region_dict to store locations of CDRs & FRs based on scheme provided
    region_dict = {}
    label_length = 0
    for index, val in enumerate(scheme_cdr_labels):
        current_label = val
        future_label = scheme_cdr_labels[index + 1] if index + 1 < len(scheme_cdr_labels) else None
        if current_label == future_label:
            label_length += 1
        else:
            region_dict[current_label] = {'index_start' : index - label_length, 'index_end' : index + 1} # Index not included in end
            label_length = 0
    
    return aligned_seqs_list, region_dict, scheme_cdr_labels
