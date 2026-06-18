from Bio import SeqIO
from antpack import SingleChainAnnotator
import pandas as pd
import os

def get_seqs_from_fasta(fasta_path: str, suffix_split: str = None, seq_type: str = None, anti: bool = False):
    """ Function to return a dictionary of sequence IDs and sequences from a fasta file
    Args:
        fasta_path (str): path to fasta file containing sequences
    Returns:
        dict: dictionary of sequence IDs and sequences
    """
    if anti: # Extract both heavy and light chain sequences for antibodies
        heavy_seqs = {}
        light_seqs = {}
        anti_ids = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = '_'.join(record.id.split("_")[0:2])
            if "heavy" in record.id:
                heavy_seqs[seq_id] = str(record.seq)
            elif "light" in record.id:
                light_seqs[seq_id] = str(record.seq)
            anti_ids.append(seq_id)
        unique_antibodies = list(set(anti_ids))

        seqs = {}
        for anti_id in unique_antibodies:
            seqs[anti_id] = heavy_seqs[anti_id] + light_seqs[anti_id]
    
    else: # Extract single chain sequences
        seqs = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            record_id_split = record.id.split(suffix_split)[0] if suffix_split else record.id
            record_id_final = f"{record_id_split}_{seq_type}" if seq_type else record_id_split
            seqs[record_id_final] = str(record.seq)
    
    return seqs