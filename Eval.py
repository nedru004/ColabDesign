import numpy as np
import pandas as pd
import Levenshtein
import blosum as bl
from ColabDesign.Scfv import Scfv

class Eval:
    """ Class to evaluate designed ScFv sequences against target sequences
    """
    def __init__(self, mpnn_csv_path: str, ref_anti_fasta_path: str, ref_scfv_fasta_path: str, linker_dict: dict, orientation_dict: dict, scheme = 'martin'):
        """ Initialization function for Eval class
        Args:
            mpnn_csv_path (str): Path to CSV file containing designed scFv sequences from MPNN
            og_anti_fasta_path (str): Path to FASTA file containing original antibody sequences
            ref_scfv_fasta_path (str): Path to FASTA file containing reference scFv sequences
            scheme (str): Numbering scheme to use for scFv or Paired (Heavy + Light) annotation (default: 'martin')
        """         
        self.mpnn_csv_path = mpnn_csv_path
        self.ref_anti_fasta_path = ref_anti_fasta_path
        self.ref_scfv_fasta_path = ref_scfv_fasta_path
        self.ref_paired_seq_annotator = Scfv(scheme=scheme)
        self.ref_scfv_seq_annotator = Scfv(scheme=scheme)
        self.linker_dict = linker_dict
        self.orientation_dict = orientation_dict
    
    def load_refs(self, target_dict: dict):
        """ Function to load reference scFv sequences from FASTA file
        Returns:
            dict: Dictionary with keys being scFv IDs and values being sequences
        """ 
        # Load reference scFv sequences        
        ref_scfv_seq_dict, ref_scfv_seq_ids = self.ref_scfv_seq_annotator.extract_seqs(self.ref_scfv_fasta_path, suffix_split = "_-_", seq_type = "", anti = False)

        # Load original antibody sequences
        ref_paired_seq_dict, ref_paired_seq_ids = self.ref_paired_seq_annotator.extract_seqs(self.ref_anti_fasta_path, suffix_split = "", seq_type = "", anti = True)

        # Annotate reference sequences
        self.annotated_ref_paired_seqs = self.ref_paired_seq_annotator.annotate_seqs(self.linker_dict, self.orientation_dict, target_dict= {}, generate_motif_commands = False)
        self.annotated_ref_scfv_seqs = self.ref_scfv_seq_annotator.annotate_seqs(self.linker_dict, self.orientation_dict, target_dict= target_dict, generate_motif_commands = True)
        # Combine annotated paired and scFv sequences
        annotated_ref_seqs = {**self.annotated_ref_paired_seqs, **self.annotated_ref_scfv_seqs}
        self.annotated_ref_seqs = annotated_ref_seqs
        return self.annotated_ref_seqs

    def load_designs(self, df_designs: pd.DataFrame, ref_design_name: str = ''):
        """ Function to load designed scFv sequences from MPNN CSV file
        Returns:
            dict: Dictionary with keys being scFv IDs and values being sequences
        """ 
        # Extract orientation, location of heavy FRs & CDRs, and location of light FRs & CDRs from reference name
        if ref_design_name != '':
            self.ref_name = ref_design_name
        else:
            raise ValueError("Reference design scfv ID must be provided to load designs.")
        
        for ref_name, orientation in self.orientation_dict.items():
            if self.ref_name in ref_name:
                self.design_orientation = orientation
                self.design_heavy_region_loc_dict = self.annotated_ref_seqs[ref_name]['heavy']['region_loc_dict']
                self.design_light_region_loc_dict = self.annotated_ref_seqs[ref_name]['light']['region_loc_dict']
                break
        
        # Depending on design orientation, set how columns are named
        if self.design_orientation == 'VH-VL':
            col_names = {0 : 'heavy_seq', 1: 'light_seq'}
        elif self.design_orientation == 'VL-VH': # 'VL-VH'
            col_names = {0 : 'light_seq', 1: 'heavy_seq'}

        # Load designed scFv sequences from MPNN CSV
        if df_designs is None:
            df_designs = pd.read_csv(self.mpnn_csv_path, index_col=0)
        else:
            df_designs = df_designs.copy()
        df_designs['scfv_seq'] = df_designs['seq'].str.split('/').str[0]
        df_seqs = df_designs['scfv_seq'].str.split(self.linker_dict[self.ref_name], expand=True).rename(columns=col_names)
        df_designs = pd.concat([df_designs, df_seqs], axis=1)
        # Sort by RFDiffusion Design First and then PTM from best to worst
        df_designs.sort_values(by = ['rmsd', 'ptm'], ascending = [True, False], inplace = True)
        # Convert DataFrame to dictionary
        # If seq_id column is present, include it in df_designs_dict
        if 'seq_id' in df_designs.columns:
            df_designs_dict = df_designs[['seq_id', 'design', 'n', 'ptm', 'scfv_seq', 'heavy_seq', 'light_seq']].to_dict(orient='records')
        else:
            df_designs_dict = df_designs[['design', 'n', 'ptm', 'scfv_seq', 'heavy_seq', 'light_seq']].to_dict(orient='records')

        
        
        # Create design scFv dictionary. Need to do this manually as AntPack does not correctly annotate designed sequences. Hints that designed seqs may not be valid and are not antibody-like
        annotated_design_seqs = {}
        for record in df_designs_dict:
            # Extract scFv ID, heavy and light sequences from record
            ptm_str = f"{record['ptm']}"[:5].split('.')[1]
            scfv_id = f"RF_design{record['design']}_num{record['n']}_ptm_{ptm_str}"
            heavy_seq = record['heavy_seq']
            light_seq = record['light_seq']

            # Initialize design scFv entry
            if 'seq_id' in record:
                scfv_id = f"{record['seq_id']}"
            else:
                scfv_id = scfv_id
            
            annotated_design_seqs[scfv_id] = {'heavy' : {}, 'light' : {}, 'seq' : '', 'orientation' : '', 'linker' : ''}
            
            # Annotate designed scFv sequences chain independent properties
            annotated_design_seqs[scfv_id]['orientation'] = self.design_orientation
            annotated_design_seqs[scfv_id]['linker'] = self.linker_dict[self.ref_name]

            # Annotate designed scFv sequences chain dependent properties: region locations, region sequences, chain seq
            annotated_design_seqs[scfv_id]['heavy']['region_loc_dict'] = self.design_heavy_region_loc_dict
            annotated_design_seqs[scfv_id]['light']['region_loc_dict'] = self.design_light_region_loc_dict
            annotated_design_seqs[scfv_id]['heavy']['seq'] = heavy_seq
            annotated_design_seqs[scfv_id]['light']['seq'] = light_seq
            annotated_design_seqs[scfv_id]['heavy']['region_seqs_dict'] = {index : heavy_seq[locs['start']:locs['end']+1] for index, locs in self.design_heavy_region_loc_dict.items()}
            annotated_design_seqs[scfv_id]['light']['region_seqs_dict'] = {index : light_seq[locs['start']:locs['end']+1] for index, locs in self.design_light_region_loc_dict.items()}
            
            # Add full scFv sequence
            annotated_design_seqs[scfv_id]['seq'] = record['scfv_seq']
        
        self.annotated_design_seqs = annotated_design_seqs
        return self.annotated_design_seqs
    #============================== Evaluation Helper Function ======================#
    #============================== Structural Evaluation Approach ==================#
    def generate_pdb_region_ranges(self, anti_name: str) -> dict:
        """
        Function to generate list of FR and CDR 1-indexed ranges for an annotated antibody and specific chain,
        Use as input to align specific reference and design PDBs
        Args:
            anti_name: (str) name of antibody
        Return:
            anti_pdb_dict = {'heavy' : {'fmwk' : fr_range, 'cdr' : cdr_range,
                             'light' : {'fmwk' : fr_range, 'cdr' : cdr_range,
                             'linker' : linker_range}}}
            fr_range: list of 1-indexed tuples defining start and end (both inclusive) of FR1-FR4
            cdr_range: list of 1-indexed tuples defining start and end (both inclusive) of CDR1-CDR3
            linker_range: list of 1-indexed tuple defining start and end (both inclusive) of linker
        """
        for ref_id in self.annotated_ref_seqs:
            if anti_name in ref_id:
                anti_dict = self.annotated_ref_seqs[anti_name]
        
        linker_length = len(anti_dict['linker'])
        orientation = anti_dict['orientation']
        
        # Have to apply an offset depending on which chain is first
        if orientation == 'VH-VL':
            first_chain = 'heavy'
            second_chain = 'light'
            is_scfv = True
        elif orientation == 'VL-VH':
            first_chain = 'light'
            second_chain = 'heavy'
            is_scfv = True
        elif orientation == 'unknown':
            is_scfv = False
    
        # Compute Linker PDB Positions if want to be used for alignment or measurement. Only for scFvs
        if is_scfv:
            first_chain_length = len(anti_dict[first_chain]['seq'])
            linker_range = [(first_chain_length + 1, first_chain_length + linker_length)]

        # Create respective heavy or light chain PDB FMWK & CDR residue lists
        anti_pdb_dict = {}
        for chain in ['heavy', 'light']:
            fr_range = []
            cdr_range = []
            anti_reg_loc_dict = anti_dict[chain]['region_loc_dict'] # Extract 0-indexed dictionary of FR & CDR start and end indices
            
            # Adjustment to account for linker
            if is_scfv:
                if chain == first_chain:
                    adj = 0
                elif chain == second_chain:
                    adj = len(anti_dict[first_chain]['seq']) + linker_length # adj is based on first chain length & linker
            else:
                adj = 0
            for reg_id, index_dict in anti_reg_loc_dict.items():
                index_start_one_indexed = index_dict['start'] + 1 + adj
                index_end_one_indexed = index_dict['end'] + 1 + adj
                index_tuple = (index_start_one_indexed, index_end_one_indexed)
        
                if "fmwk" in reg_id:
                    fr_range.append(index_tuple)
        
                elif "cdr" in reg_id:
                    cdr_range.append(index_tuple)
            
            anti_pdb_dict[chain] = {'fmwk' : fr_range, 'cdr' : cdr_range}
        
        # Add in Linker Location in PDB
        if is_scfv:
            anti_pdb_dict['linker'] = linker_range

        return anti_pdb_dict
    #============================== EVALUATION METRICS ==============================#
    def compute_levenshtein(self, seq1: str, seq2: str, similarity: bool = True):
        """ Function to compute Levenshtein distance or similarity between two sequences
        Args:
            seq1 (str): First sequence
            seq2 (str): Second sequence
            similarity (bool): If True, compute similarity instead of distance (default: False)
        Returns:
            int or float: Levenshtein distance or similarity between the two sequences
        """ 
        if seq1 is not None and seq2 is not None:
            distance = Levenshtein.distance(seq1, seq2)
            if similarity:
                max_len = max(len(seq1), len(seq2))
                similarity = (1 - (distance / max_len)) * 100 # Return similarity as a Percent (Better for Visualizations)
                return similarity
            else:
                return distance
    
    def compute_sequence_identity(self, seq1: str, seq2: str):
        """ Function to compute sequence identity between two sequences
        Args:
            seq1 (str): First sequence
            seq2 (str): Second sequence
        Returns:
            float: Sequence identity between the two sequences
        """ 
        if seq1 is not None and seq2 is not None:
            matches = sum(a == b for a, b in zip(seq1, seq2))
            identity = matches / max(len(seq1), len(seq2))
            return identity
    
    def compute_blosum_score(self, seq1: str, seq2: str, matrix: int):
        """ Function to compute BLOSUM score between two sequences
        Args:
            seq1 (str): First sequence
            seq2 (str): Second sequence
            matrix (int): BLOSUM matrix to use (e.g., 62, 80, 100)
        Returns:
            int: BLOSUM score between the two sequences
        """ 
        blosum_matrix = bl.BLOSUM(matrix)
        print(f"Using BLOSUM{matrix} matrix for scoring. with seq1: {seq1} and seq2: {seq2}")
        if seq1 is not None and seq2 is not None: # Ensure sequences are not None or empty
            if len(seq1) != len(seq2):
                raise ValueError("Sequences must be of equal length to compute BLOSUM score.")
            # Iterate through each region's sequence and compute BLOSUM score for entire region as sum of position-wise scores
            blosum_reg_score = 0
            for index, amino_acid in enumerate(seq1):
                seq1_aa = amino_acid
                seq2_aa = seq2[index]
                blosum_pos_score = blosum_matrix[seq1_aa][seq2_aa]
                blosum_reg_score += blosum_pos_score
            
        return blosum_reg_score
    
    def compute_region_metrics(self, ref_seq_dict: dict, design_seq_dict: dict, metric: str = 'levenshtein'):
        """ Function to compute evaluation metrics for each region between reference and designed sequences
        Args:
            ref_seq_dict (dict): Dictionary containing annotated reference sequences
            design_seq_dict (dict): Dictionary containing annotated designed sequences
            metric (str): Metric to compute ('levenshtein', 'identity', 'blosum')
        Returns:
            dict: Dictionary containing computed metrics for each region
        """

        chain_types = ['heavy', 'light']
        region_metrics_dict = {}
        for chain in chain_types:
            print(f"Computing {metric} metrics for {chain} chain...")

            # Initialize region metrics dictionary
            region_metrics_dict[chain] = {}

            # Extract chain region specific seqs
            ref_region_seqs_dict = ref_seq_dict[chain]['region_seqs_dict']
            design_region_seqs_dict = design_seq_dict[chain]['region_seqs_dict']

            # Iterate through each region and compute metrics
            for region_index in ref_region_seqs_dict.keys():
                print(f"  Processing region: {region_index}")
                ref_region_seq = ref_region_seqs_dict.get(region_index, None)
                design_region_seq = design_region_seqs_dict.get(region_index, None)

                if metric == 'levenshtein':
                    region_metric = self.compute_levenshtein(ref_region_seq, design_region_seq)
                elif metric == 'identity':
                    region_metric = self.compute_sequence_identity(ref_region_seq, design_region_seq)
                elif metric == 'blosum':
                    region_metric = self.compute_blosum_score(ref_region_seq, design_region_seq, matrix=62)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

                region_metrics_dict[chain][region_index] = region_metric
        
        return region_metrics_dict
    
    def compute_metrics_ref_paired(self, metrics: list, ref_name : str):
        """ Function to compute metrics for all designed seqs and ref design seqs vs the original paired Ab seq
        Args:
            metrics: list -> List of metrics used to compare the scfv seqs to the original paired Ab seq
        Returns:
            dict: Dictionary containing Blosum62 scores for all designed seqs and ref design seqs vs the original paired Ab seq
        """
        metric_ref_paired_scores = {metric : {} for metric in metrics} # Adjusting definition since using metrics (list input)

        # Prepare reference and design sequence dictionaries
        for key in self.annotated_ref_seqs.keys():
            if ref_name in key: # Indicates ref name is part of the key for either the ref scfv or ref paired seq
                if 'manod' in key or 'scfv' in key:
                    ref_design_key = key
                elif ref_name == key:
                    ref_paired_key = key

        # Get correct reference paired Ab sequence dictionary & merge annotated ref scFv & designed scfv seq dicts
        ref_ab_seq_dict = self.annotated_ref_paired_seqs[ref_paired_key]
        ref_design_seq_dict = {ref_design_key : self.annotated_ref_scfv_seqs[ref_design_key]}
        ref_design_seq_dict = self.annotated_design_seqs | ref_design_seq_dict


        for scfv_id, design_seq_dict in ref_design_seq_dict.items():

            # Iterate through each metric in the metrics list
            for metric in metrics:

                region_metric = self.compute_region_metrics(ref_ab_seq_dict, design_seq_dict, metric= metric)
                metric_ref_paired_scores[metric][scfv_id] = region_metric
        
        return metric_ref_paired_scores
    
    def compute_metrics_ref_paired_scfv(self, metrics: list = ['levenshtein', 'identity'], ref_name : str = ""):
        """ Function to compute metrics for all designed seqs vs the ref paired ab seq and ref scfv seq
            May need to adjust this so its just against the ref scfv seq. Do not see value in comparing scfvs to paired Ab
        Args:
            metrics: list of metrics referring to evaluation metrics previously defined
        Returns:
            dict: Nested dictionary of {metric: {scfv_design_id : metric_score}}
        """

        # Prepare reference and design sequence dictionaries
        for key in self.annotated_ref_seqs.keys():
            if ref_name in key: # Indicates ref name is part of the key for either the ref scfv or ref paired seq
                if 'manod' in key or 'scfv' in key:
                    ref_design_key = key
                elif ref_name == key:
                    ref_paired_key = key
        
        # Create reference (paired and ref_scfv) and design (RFDiffusion + MPNN, and ref_scfv) sequence dictionaries
        ref_seqs_dict = {ref_paired_key : self.annotated_ref_paired_seqs[ref_paired_key],
                                  ref_design_key : self.annotated_ref_scfv_seqs[ref_design_key]}
        
        ref_design_seq_dict = {ref_design_key : self.annotated_ref_scfv_seqs[ref_design_key]}
        ref_design_seq_dict = self.annotated_design_seqs | ref_design_seq_dict

        # Create metric-specific score dictionaries with the key being respective ref (paired or ref_scfv) id
        metric_ref_all_scores = {}
        for ref_id, ref_seq_dict in ref_seqs_dict.items(): # Iterate through both baseline paired and scfv seqs
            
            metric_ref_scores = {metric : {} for metric in metrics} # Initialize empty dict with key being each metric
            
            for scfv_id, design_seq_dict in ref_design_seq_dict.items(): # Iterate through all the design seqs

                for metric in metrics: # Iterate through all the metrics

                    region_metric = self.compute_region_metrics(ref_seq_dict, design_seq_dict, metric = metric)
                    metric_ref_scores[metric][scfv_id] = region_metric
            
            metric_ref_all_scores[ref_id] = metric_ref_scores # Assign metric-specific scores to the key being respective baseline
        
        return metric_ref_all_scores




    

            
    





    
