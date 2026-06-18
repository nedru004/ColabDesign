import sys
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

class MutLocPredictor():
    """ Class to predict mutable locations in scFv based on scfv-target docked complex and FR & CDR residue mapping.
    Args:
        one_indexed_chain_residue_mapping (dict): Mapping of heavy, light chain & linker residues with 1-indexed positions.
        docked_pdb_file_path (str): Path to the docked PDB file.
        scfv_chain_id (str): Chain ID of the scFv in the PDB file. Default is 'A'.
    Returns:
        Methods to predict mutable locations, extract sequences, and generate motif scaffolding commands.
        Primary method: predict_mut_loc
    
    Code adapted from https://github.com/jbderoo/scFv_Pmpnn_AF2/blob/main/scripts/loops_from_sequence.py
    """
    def __init__(self, one_indexed_chain_residue_mapping: dict, docked_pdb_file_path: str, scfv_chain_id: str = 'A'):
        self.chain_residue_mapping = one_indexed_chain_residue_mapping
        self.docked_pdb_file_path = docked_pdb_file_path
        self.scfv_chain_id = scfv_chain_id
    
    def pdb_to_sequence_manual(self, pdb_path):
        """Manually extracts the amino acid sequence from a PDB file."""
        # Define a dictionary to convert three-letter codes to one-letter codes
        three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O'  # Including selenocysteine (Sec) and pyrrolysine (Pyl)
        }

        sequences = {}
        last_residue_id = None

        try:
            with open(pdb_path, 'r') as file:
                for line in file:
                    if line.startswith("ATOM"):
                        chain_id = line[21]
                        residue_name = line[17:20].strip()
                        residue_id = line[22:27].strip()  # Including insertion code

                        if chain_id not in sequences:
                            sequences[chain_id] = ''

                        if residue_id != last_residue_id:
                            if residue_name in three_to_one:
                                sequences[chain_id] += three_to_one[residue_name]
                            else:
                                sequences[chain_id] += '?'  # Unrecognized residue
                        last_residue_id = residue_id

        except Exception as e:
            return f"Error reading PDB file: {e}"

        return sequences


    def parse_pdb(self, pdb_path):
        """ Parse PDB to extract coordinates and other data per atom. """
        atoms = []
        with open(pdb_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    atom_data = {
                        'chain': line[21],
                        'res_id': line[22:26].strip() + line[21],  # Combining residue number with chain identifier directly
                        'res_name': line[17:20].strip(),
                        'x': float(line[30:38]),
                        'y': float(line[38:46]),
                        'z': float(line[46:54])
                    }
                    atoms.append(atom_data)
        return atoms


    # Function to parse the PDB file and extract chain information
    def parse_pdb_file(self, file_path):
        chains = defaultdict(list)
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    chain_id = line[21]
                    residue_number = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    chains[chain_id].append((residue_number, (x, y, z)))
        return chains

    def fill_gaps_and_remove_isolated_residues(self, residues):
        # Start with the original list of residues in contact
        filled_residues = sorted(set(residues))

        # Initialize a list to hold the intermediate set of residues, including gap-filled ones
        intermediate_residues = []

        # Go through the sorted list and fill in the gaps
        i = 0
        while i < len(filled_residues) - 1:
            intermediate_residues.append(filled_residues[i])

            # Check the gap between the current and next residue
            gap = filled_residues[i + 1] - filled_residues[i]

            # If the gap is 2 or 3 (indicating 1 or 2 residues missing between them), fill it
            if gap <= 3:
                # Add the missing residues to the intermediate list
                intermediate_residues.extend(range(filled_residues[i] + 1, filled_residues[i + 1]))

            i += 1

        # Add the last residue since it's not covered in the loop
        intermediate_residues.append(filled_residues[-1])

        # Remove isolated residues
        final_residues = []
        for res in intermediate_residues:
            # Check if the residue has neighbors within 5 residue numbers
            has_neighbors = any(abs(res - other) <= 3 for other in intermediate_residues if other != res)
            if has_neighbors:
                final_residues.append(res)

        return sorted(final_residues)

    def find_close_residues(self, pdb_path, residues, cutoff=3.0):
        """
        Find neighboring residues within a given cutoff distance from a list of input residues.

        Args:
            pdb_file_path (str): Path to the PDB file.
            residues (list): List of residue IDs (e.g., ['47A']) to find neighbors for.
            cutoff (float): Maximum distance (in Angstroms) to consider a residue as a neighbor.

        Returns:
            list: List of unique residue numbers that are within the cutoff distance of any residue in the input list.
        """
        coords = []
        res_ids = []
        chains = []

        with open(pdb_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    chain = line[21]
                    res_id = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    coords.append([x, y, z])
                    res_ids.append(res_id)
                    chains.append(chain)

        coords = np.array(coords)
        res_ids = np.array(res_ids)
        chains = np.array(chains)

        input_chains = [res[-1] for res in residues]
        input_res_ids = [int(res[:-1]) for res in residues]

        neighboring_residues = set()

        for res_id, chain in zip(input_res_ids, input_chains):
            res_mask = (res_ids == res_id) & (chains == chain)
            res_coords = coords[res_mask]
            if res_coords.ndim == 1:
                res_coords = res_coords.reshape(1, -1)

            other_res_mask = ~res_mask
            other_res_coords = coords[other_res_mask]
            other_res_ids = res_ids[other_res_mask]
            other_chains = chains[other_res_mask]

            if other_res_coords.ndim == 1:
                other_res_coords = other_res_coords.reshape(1, -1)

            pairwise_dists = cdist(res_coords, other_res_coords)
            neighboring_residues.update([other_res_id for other_res_id, other_chain, dists in zip(other_res_ids, other_chains, pairwise_dists.T) if np.any(dists <= cutoff) and other_chain in input_chains])

        output_residues = [res_id for res_id in neighboring_residues]

        return sorted(output_residues)

    def get_longest_chain_length(self, pdb_path : str, scfv_chain_id : str):
        """Returns the length of the longest chain in a PDB file by counting unique residues."""
        chain_residues = {}

        try:
            with open(pdb_path, 'r') as file:
                for line in file:
                    if line.startswith("ATOM"):  # Ensures we only process ATOM lines
                        chain_id = line[21]
                        residue_id = line[22:26].strip() + line[26]  # Combining residue number with insertion code

                        if chain_id not in chain_residues:
                            chain_residues[chain_id] = set()
                        
                        chain_residues[chain_id].add(residue_id)
        
        except FileNotFoundError:
            return "PDB file not found."

        # Find the longest chain by comparing the size of residue sets
        longest_chain_length = len(chain_residues[scfv_chain_id])
        longest_chain_id = scfv_chain_id


        return longest_chain_length, longest_chain_id

    def extract_sequence_from_pdb(pdb_file_path):

        three_to_one = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
        }


        sequences = {}
        seen_residues = set()

        with open(pdb_file_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM"):
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_seq = line[22:26].strip()
                    res_id = (chain_id, res_seq)

                    if res_id not in seen_residues:
                        amino_acid = three_to_one[res_name]
                        if amino_acid:
                            if chain_id not in sequences:
                                sequences[chain_id] = []
                            sequences[chain_id].append(amino_acid)
                            seen_residues.add(res_id)

        # Join sequences of each chain with a colon separator
        final_sequence = ':'.join(''.join(sequences[chain_id]) for chain_id in sorted(sequences.keys()))

        return final_sequence

    def convert_residue_dict_list(self, chain_residue_dict: dict):
        """
        Extract CDR residue 1-indexed positions from chain_residue_dict and store in a list 
        """

        unsorted_loops = []
        for _, fr_cdr_dict in chain_residue_dict.items():
            if type(fr_cdr_dict) == dict:
                cdr_one_indexed_tuple_list = fr_cdr_dict['cdr']
                for one_indexed_tuple in cdr_one_indexed_tuple_list:
                    cdr_start, cdr_end = one_indexed_tuple
                    cdr_pos = list(range(cdr_start, cdr_end + 1)) # Last index not included in range
                    unsorted_loops.extend(cdr_pos)
            elif type(fr_cdr_dict) == list:
                linker_tuple = fr_cdr_dict[0]
                linker_start, linker_end = linker_tuple
                linker_pos = list(range(linker_start, linker_end + 1))
        
        all_cdr_pos = sorted(unsorted_loops) # Sort in ascending order
        return all_cdr_pos, linker_pos

    def generate_motif_scaffolding_command(self, all_residues: list, fixed_residues: list, target: str = ""):
        """ Generate motif scaffolding command for AF2-multimer
            Args:
                fixed_residues: List of 1-indexed residue positions to be fixed during design
                all_residues: List of all 1-indexed residue positions in the scFv chain
            Returns:
                motif_scaffolding_command: str
        """
        motif_scaffolding_command = ""
        fixed_pos = []
        free_residues = []
        chain_id = 'A'
        for res_pos in all_residues: # Starting from 1 to length of scfv
        
            if res_pos not in fixed_residues: # Accounts for vernier zone residues, CDRs, and the linker
                fixed_pos.append(res_pos)

            elif res_pos in fixed_residues: # Free Residue
                if fixed_pos != []: # Only occurrs if just hit the first FR residue in the set of continuous FR residues subset
                
                    if free_residues != []: # Implies there are free residues and need to quickly add to the motif scaffolding command before reset
                        free_pos_start = min(free_residues)
                        free_pos_end = max(free_residues)
                        distance = free_pos_end - free_pos_start
                        motif_command_subset = f"{distance + 1}/"
                        motif_scaffolding_command += motif_command_subset 
                
                    # Code block for adding in the fixed motif command subset since just hit the first Free Residue
                    fixed_pos_start = min(fixed_pos)
                    fixed_pos_end = max(fixed_pos)
                    motif_command_subset = f"{chain_id}{fixed_pos_start}-{fixed_pos_end}/"
                    motif_scaffolding_command += motif_command_subset

                    # Reset both free and fixed residues list
                    fixed_pos = []
                    free_residues = []
                    free_residues.append(res_pos) # First Framework Position
                elif fixed_pos == []:
                    free_residues.append(res_pos)
        # Handle the last set of FR residues after exiting the loop
        final_fixed_residue_pos = int(motif_scaffolding_command.split('/')[-2].split('-')[1])
        remaining_free_distance = len(all_residues) - final_fixed_residue_pos
        motif_scaffolding_command_final = motif_scaffolding_command + f"{remaining_free_distance}"

        if target != "": # Append target at the end of the motif scaffolding command if specified
            motif_scaffolding_command_final = motif_scaffolding_command_final + f"/0 {target}"
        
        return motif_scaffolding_command_final


    def predict_mut_loc(self, output: str, verbose: bool = True, simple_grab: bool = False, neighbor_cutoff_distance: int = 3):
        """ Using the mapping and location of CDR & Linker residues along with the docked pdb file
            and initially defined neighborhood cutoff distance.
            Identify all Vernier Zone residues (FR residues critical for ensuring proper orientation and stable display of binding interface)
            Identify all mutable FR Residues: Residues which are not of type: Linker, Vernier Zone, and CDR
            
            Args:
                output: -> 'cdrs', 'framework', 'vernier_cdrs'
                verbose: print output with extra context
                simple_grab: True -> Don't identify residues within 5 of CDR position as unmutable
                neighborhood_cutoff_distance: cutoff distance for determining Vernier or not from CDR-epitope complex
            
            Return user-specified output: 
                'cdrs' -> CDR positions
                'framework' -> Non-CDR, Linker, and Vernier Zone residues positions
                'vernier_cdrs' -> Vernier Zone and CDR residue positions 
        """
        pdb_path = self.docked_pdb_file_path
        cdr_pos, linker_pos = self.convert_residue_dict_list(chain_residue_dict = self.chain_residue_mapping)
        scfv_length, scfv_id = self.get_longest_chain_length(pdb_path= pdb_path, 
                                                        scfv_chain_id= self.scfv_chain_id)
        cdr_chain_pos = []
        [cdr_chain_pos.append(f'{x}{scfv_id}') for x in cdr_pos]
        residues = self.find_close_residues(pdb_path = pdb_path, residues = cdr_chain_pos, cutoff = neighbor_cutoff_distance)

        if simple_grab == True:
            vernier = residues
        else:
            vernier = self.fill_gaps_and_remove_isolated_residues(residues)
        
        vernier_and_loops = [x for x in vernier if x not in linker_pos]
        all_resi = list(range(1, scfv_length + 1))
        framework = [x for x in all_resi if x not in vernier_and_loops and x not in linker_pos]

        if output == 'cdrs':
            fixed_residues = cdr_pos
        elif output == 'framework':
            fixed_residues = framework
        elif output == 'vernier_cdrs':
            fixed_residues = vernier_and_loops

        if verbose:
            print(f"CDR positions: {cdr_pos}")
            print(f"Framework positions: {framework}")
            print(f"Vernier and CDR positions: {vernier_and_loops}")
        
        motif_scaffolding_command = self.generate_motif_scaffolding_command(all_residues= all_resi, fixed_residues= fixed_residues)
        print(f"Motif Scaffolding Command: {motif_scaffolding_command}")
        
        if output == 'cdrs':
            return cdr_pos, motif_scaffolding_command
        elif output == 'framework':
            return framework, motif_scaffolding_command
        elif output == 'vernier_cdrs':
            return vernier_and_loops, motif_scaffolding_command