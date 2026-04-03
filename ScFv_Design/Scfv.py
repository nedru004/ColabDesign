from hmac import new
import numpy as np
import antpack
from antpack import PairedChainAnnotator
from Bio import SeqIO
from ColabDesign.general_utils import get_seqs_from_fasta
import os
import sys

class Scfv:
    def __init__(self, scheme: str = "martin"):
        self.scheme = scheme
        self.scfv_dict = {}
        self.pc_annotator = PairedChainAnnotator(scheme = scheme)
    
    def extract_seqs(self, fasta_path,  suffix_split: str = "_-_", seq_type: str = "", anti = False):
        """ Function to extract heavy and light chain sequences from scfv sequences in fasta file
        Args:
            suffix_split (str): Suffix to split fasta record IDs
            seq_type (str): Suffix to append to sequence IDs
        Returns:
            dict: nested dictionary of scfv sequences with heavy and light chains
        """ 
        scfv_dict = get_seqs_from_fasta(fasta_path = fasta_path, suffix_split= suffix_split, seq_type = seq_type, anti = anti)
        scfv_ids = list(scfv_dict.keys())
        print(scfv_dict)
        # Only set scfv_dict and scfv_ids if they are not already set, allows for updating later
        if self.scfv_dict is None or len(self.scfv_dict) == 0:
            self.scfv_dict = scfv_dict
            self.scfv_ids = scfv_ids
        else: # If already set, update with new sequences
            self.scfv_dict.update(scfv_dict)
            self.scfv_ids = list(self.scfv_dict.keys())

        return self.scfv_dict, self.scfv_ids
    
    def update_seqs(self, seq_dict: dict = {}, seq_id: str = "", seq: str = ""):
        """ Function to update scfv sequences with new sequences stored in a dictionary
        Args:
            seq_dict (dict): Dictionary with keys being scFv or paired seq IDs and values being sequences (heavy + linker + light or vice versa or regular paired sequences)
            seq_id (str): ID of scFv or paired sequence to update
            seq (str): New scFv seq or paired sequence
        Returns:
            dict: updated dictionary of scfv and/or paired sequences with sequences
        """ 
        # If no new_seq_dict is provided, use seq_id and seq to create a new dictionary
        if seq_dict == {}:
            seq_dict = {seq_id : seq} if seq_id != "" and seq != "" else seq_dict
        
        # If seq_dict, seq_id, and seq are all not provided, raise error
        elif seq_dict == {} and seq_id == "" and seq == "":
            raise ValueError("No new sequences provided to update scfv_dict with")
        
        # Only set scfv_dict and scfv_ids if they are not already set, allows for updating later
        if self.scfv_dict is None or len(self.scfv_dict) == 0:
            self.scfv_dict = seq_dict
            self.scfv_ids = list(self.scfv_dict.keys())
        else: # If already set, update with new sequences
            self.scfv_dict.update(seq_dict)
            self.scfv_ids = list(self.scfv_dict.keys())
        return self.scfv_dict

    def annotate_seqs(self, linker_dict, orientation_dict: dict, target_dict: dict, generate_motif_commands: bool = True):
        """ Function to annotate scfv sequences into heavy and light chains using antpack's PairedChainAnnotator
        Returns:
            dict: nested dictionary of scfv sequences with annotated heavy and light chains
        """
        annotated_scfv_seqs = {}
        for scfv_id, scfv_seq in self.scfv_dict.items():
            
            heavy_dict, light_dict = self.get_cdr_fr_labels(paired_chain = scfv_seq)
            complete_heavy_dict = self.get_region_loc_dict(chain_dict = heavy_dict)
            complete_light_dict = self.get_region_loc_dict(chain_dict = light_dict)
            
            annotated_scfv_seqs[scfv_id] = {
                'heavy' : complete_heavy_dict,
                'light' : complete_light_dict,
                'linker' : linker_dict.get(scfv_id, ''),
                'seq' : scfv_seq,
                'orientation' : orientation_dict.get(scfv_id, "unknown")
            }
            
            target_command_subset = target_dict.get(scfv_id, '')
            print(f"Generated Annotations for {scfv_id}, about to begin motif scaffolding command generation...")
            if generate_motif_commands:
                annotated_scfv_seqs[scfv_id]['motif_scaffolding_command'] = self.generate_motif_scaffolding_contig(scfv_id, scfv_annotated_dict = annotated_scfv_seqs,
                                                                                                                   target_command_subset= target_command_subset)
            print(f"Annotation done for {scfv_id}")
        
        self.annotated_scfv_seqs = annotated_scfv_seqs
        return self.annotated_scfv_seqs

    def get_cdr_fr_labels(self, paired_chain:str):
        """ Function to generate heavy and light FR & CDR labels from annotated scfv sequences
        Args:
            paired_chain (str): scfv sequence with heavy and light chains & linker (Orientation independent)
        Returns:
            tuple: heavy chain labels list, light chain labels list for given scfv sequence
        """

        # Generate Heavy and Light Alignments
        hv_alignment, lv_alignment = self.pc_annotator.analyze_seq(paired_chain)

        # Extract untrimmed numbering, percent identity, chain ID, and error message for both chains
        hv_numbering, hv_percent_identity, hv_chain_id, hv_error_message = hv_alignment
        lv_numbering, lv_percent_identity, lv_chain_id, lv_error_message = lv_alignment

        # Trim alignments to get heavy and light variable regions and their respective trimmed alignments
        heavy_chain, hv_trimmed_alignment, _, _ = self.pc_annotator.trim_alignment(paired_chain, hv_alignment)
        light_chain, lv_trimmed_alignment, _, _ = self.pc_annotator.trim_alignment(paired_chain, lv_alignment)

        # Assign CDR and FR labels based on trimmed alignments
        hv_labels = self.pc_annotator.assign_cdr_labels(hv_trimmed_alignment, hv_chain_id, scheme = self.scheme)
        lv_labels = self.pc_annotator.assign_cdr_labels(lv_trimmed_alignment, lv_chain_id, scheme = self.scheme)

        heavy_dict = {'heavy' : {'seq' : heavy_chain, 'labels' : hv_labels}}
        light_dict = {'light' : {'seq' : light_chain, 'labels' : lv_labels}}
        
        return heavy_dict, light_dict

    def get_region_loc_dict(self, chain_dict: dict):
        
        """ Function to generate dictionary with region location start & end indices & dictionary with CDR & FR seqs from CDR and FR labels
        
        Args:
            chain_dict (dict): Nested Dictionary containing {'chain_id' : {'seq' : str, 'labels' : list}} for either heavy or light chain 
        Returns:
            region_dict (dict): Dictionary containing:
                region_loc_dict: dictionary of region locations with start and end indices for respective FR & CDR regions
                region_seqs_dict: dictionary of region sequences for respective FR & CDR regions
        
        """
        chain_id = list(chain_dict.keys())[0]
        chain_seq = chain_dict[chain_id]['seq']
        chain_labels = chain_dict[chain_id]['labels'] 

        region_indices_dict = {}
        for index, label in enumerate(chain_labels):
            if label not in region_indices_dict:
                region_indices_dict[label] = []
            region_indices_dict[label].append(index)


        region_loc_dict = {index : {'start': locs[0], 'end': locs[-1]} for index, locs in region_indices_dict.items()}
        region_seqs_dict = {index : chain_seq[locs['start']:locs['end']+1] for index, locs in region_loc_dict.items()}

        region_dict = {'region_loc_dict' : region_loc_dict, 'region_seqs_dict' : region_seqs_dict, 'seq' : chain_seq}

        return region_dict
    
    def generate_motif_scaffolding_contig(self, scfv_id: str, scfv_annotated_dict: dict, target_command_subset: str) -> str:
        """ Generating motif scaffolding command for RFDiffusion, command varies based on orientation & target chain length
            Args:
                scfv_id (str): scfv identifier
                scfv_annotated_dict (dict): nested dictionary of annotated scfv sequences with heavy chain annotations, light chain annotations, linker, and orientation
            Returns:
                str: motif scaffolding contig command for RFDiffusion
         """

        # Extract information from annotated scfv dictionary
        print(scfv_id)
        linker = scfv_annotated_dict[scfv_id]['linker']
        if linker == '':
            print(f"No linker found for {scfv_id}, cannot generate motif scaffolding command")
            print("Input is probably not an scFv sequence. Verify input sequence and passed linker dictionary.")
            print("Skipping motif scaffolding command generation for this sequence...")
            return "Unable to Generate Motif Scaffolding Command due to missing linker"
        orientation = scfv_annotated_dict[scfv_id]['orientation']
        seq_len_dict = {
            'heavy' : len(scfv_annotated_dict[scfv_id]['heavy']['seq']),
            'light' : len(scfv_annotated_dict[scfv_id]['light']['seq']),
            'linker' : len(linker)
        }
        # Based on orientation define first and second chains as well as the ordering of anti dict keys and values
        anti_dict = {}
        if orientation == 'VH-VL':
            first_chain = 'heavy'
            second_chain = 'light'

        elif orientation == 'VL-VH':
            first_chain = 'light'
            second_chain = 'heavy'
        
        elif orientation == 'unknown':
            print(f"Orientation for {scfv_id} is unknown, cannot generate motif scaffolding command")
            return "Unable to Generate Motif Scaffolding Command due to unknown orientation"
        
        anti_dict[first_chain] = scfv_annotated_dict[scfv_id][first_chain]['region_loc_dict']
        anti_dict[second_chain] = scfv_annotated_dict[scfv_id][second_chain]['region_loc_dict']

        # Define some parameters upon initialization
        chain_name = 'A'
        motif_scaffolding_contig_command = ''

        # Iterate through antibody dictionary in order of the orientation previously defined
        for chain, reg_dict in anti_dict.items():

            # Have to account for linker and heavy chain length when defining positions for light chain
            if chain == first_chain:
                adj = 0
            elif chain == second_chain:
                adj = seq_len_dict['linker'] + seq_len_dict[first_chain]

            # For each cdr/fr region in a given chain:
            for reg, loc_dict in reg_dict.items():

                # For FR regions
                if "fmwk" in reg:
                    fr_length = (loc_dict['end'] - loc_dict['start']) + 1 # +1 to account for inclusive indexing
                    generate_fr_command = str(fr_length) + "/"
                    motif_scaffolding_contig_command += generate_fr_command

                # For CDR regions
                if "cdr" in reg:
                    # Account for Zero-Indexing but Command for Scaffolding is 1-indexed, keep in mind returned indices are for a range where end index is not included in slice,
                    # In our case, we do want to include end index and so if we add another +1 on top of it get an extra residue
                    cdr_start, cdr_end = loc_dict['start'] + 1 + adj, loc_dict['end'] + 1 + adj
                    generate_cdr_command = f"{chain_name}{cdr_start}-{cdr_end}/"
                    motif_scaffolding_contig_command += generate_cdr_command

            # After running through, command generation for heavy chain account for linker
            if chain == first_chain:
                # If using the length directly for start index, will be referring to final residue of heavy variable region
                motif_scaffolding_contig_command += f"A{seq_len_dict[first_chain] + 1}-{seq_len_dict[first_chain]+ seq_len_dict['linker']}/"

        # Have to add chain break as want to do motif scaffolding in presence of specific chain in target responsible for binding
        #### for CD28: its B1-118, need to modify based on target and should be a key value pair in scfv_annotated_dict-------------- not yet implemented
        if target_command_subset == '': # Will happen if only want to scaffold without target chain
            return motif_scaffolding_contig_command[:-1]  # Remove trailing slash
        else:
            motif_scaffolding_contig_command += f"0 {target_command_subset}"
            return motif_scaffolding_contig_command