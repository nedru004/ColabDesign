from Bio.PDB import PDBParser, Superimposer
from Bio.PDB import PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import numpy as np
import warnings
import py3Dmol

# Suppress warnings about atom/residue construction issues
warnings.simplefilter('ignore', PDBConstructionWarning)

class AlignPDB:
    def __init__(self, ref_pdb_path : str, design_pdb_path : str, ref_aligned_residues, ref_measured_residues, design_aligned_residues, design_measured_residues):
        self.ref_pdb_path = ref_pdb_path
        self.design_pdb_path = design_pdb_path
        self.ref_aligned_residues = ref_aligned_residues # List of tuples (start pos, end pos) 1-indexed and inclusive. Could be dict {'chain_id' : list of tuples}
        self.ref_measured_residues = ref_measured_residues # List of tuples (start pos, end pos) 1-indexed and inclusive. Could be dict {'chain_id' : list of tuples}
        self.design_aligned_residues = design_aligned_residues # List of tuples (start pos, end pos) 1-indexed and inclusive. Could be dict {'chain_id' : list of tuples}
        self.design_measured_residues = design_measured_residues # List of tuples (start pos, end pos) 1-indexed and inclusive. Could be dict {'chain_id' : list of tuples}
    
    def get_backbone_atoms(self, chain, ranges):
        """Collects C-alpha atoms for specified input ranges of 1-indexed residues positions.
           Both start and end indices are included
        """
        atoms = []
    
        for start, end in ranges:
            for i in range(start, end + 1):
                res_id = (" ", i, " ",)  # Biopython standard ID (hetero, seq_num, ins_code)
            
            try:
                residue = chain[res_id]
                # Only use C-alpha atoms for alignment
                atoms.append(residue["CA"]) 
            except KeyError:
                # Residue not found (e.g., if numbering is sparse or range is outside)
                continue
        
        return atoms

    def save_aligned_pdb(self, aligned_pdb):
        """ Function to save the aligned PDB to a specific folder in separate Inputs Folder of this repo
        """

        # Define the output file name
        save_folder_path = '/'.join(self.ref_pdb_path.split('/')[:2])
        save_full_file_path =  save_folder_path + "/" + "aligned_protein.pdb"

        io = PDBIO()
        io.set_structure(aligned_pdb.get_parent()) # Get the full structure object containing the transformed model
        io.save(save_full_file_path)

        print(f"✅ Aligned structure saved to: {save_full_file_path}")

        return save_full_file_path
    
    def get_coords_array(self, atom_list):
        """Converts a list of Biopython Atom objects into a NumPy array of coordinates."""
        return np.array([atom.get_coord() for atom in atom_list])

    def compute_rmsd(self, coords1, coords2):
        """Return rms deviations between coords1 and coords2 (assuming they are aligned)."""
        diff = coords1 - coords2
        return np.sqrt(np.sum(diff * diff) / coords1.shape[0])
    
    def visualize_aligned_proteins(self, file_a, file_b_aligned, measured_residues_indices):
        """
        Loads two aligned PDB structures and visualizes them, highlighting CDRs.
    
        Args:
            file_a (str): Path to the reference PDB
            file_b_aligned (str): Path to the superimposed PDB
            measured_residues_indices (str): Comma-separated string of residue positions whose distances were computed post-alignment between aligned structure and reference structure
                                    (e.g., '26-35, 49-66, 170-180')
        """
    
        # 1. Initialize the viewer
        view = py3Dmol.view(width=800, height=500)

        # 2. Load and Style Reference PDB
        with open(file_a, 'r') as f:
            pdb_data_a = f.read()
        view.addModel(pdb_data_a, 'pdb')
    
        # Style as translucent backbone
        view.setStyle({'model': 0}, {'cartoon': {'color': 'lightblue', 'opacity': 0.8}})
    
        # 3. Load and Style Aligned PDB
        with open(file_b_aligned, 'r') as f:
            pdb_data_b = f.read()
        view.addModel(pdb_data_b, 'pdb')
    
        # Style as opaque backbone for comparison
        view.setStyle({'model': 1}, {'cartoon': {'color': 'salmon'}})
    
        # 4. Highlight the measured_residues positions in both Reference and Aligned PDBs
        view.addStyle({'model': 0, 'resi': measured_residues_indices}, 
                  {'cartoon': {'color': 'blue'}})
        view.addStyle({'model': 0, 'resi': measured_residues_indices}, 
                  {'stick': {'colorscheme': 'blueCarbon', 'radius': 0.3}})
    
        view.addStyle({'model': 1, 'resi': measured_residues_indices}, 
                  {'cartoon': {'color': 'red'}})
        view.addStyle({'model': 1, 'resi': measured_residues_indices}, 
                  {'stick': {'colorscheme': 'redCarbon', 'radius': 0.3}})
    
        # 5. Finalize View
        view.zoomTo()
        view.show()
    def retrieve_atoms(self, pdb_model, residues):
        """ Extracting appropriate atoms based on whether its a scfv or paired antibody
        """
        # If model is an scFv:
        atoms = []
        if type(residues) == list:
            chain_scfv = pdb_model.get_chains().__next__()
            atoms = self.get_backbone_atoms(chain= chain_scfv, ranges= residues)
        
        # If model is a paired antibody
        elif type(residues) == dict:
            for chain, residues_list in residues.items():
                if chain == 'heavy':
                    pdb_chain_id = 'A'
                elif chain == 'light':
                    pdb_chain_id = 'B'
                chain = pdb_model[pdb_chain_id]
                atoms_portion = self.get_backbone_atoms(chain = chain, ranges = residues_list)
                atoms += atoms_portion
        
        return atoms

    def align_pdb(self, visualize: bool = True):
        """ Overarching function calling other functions to 
            0. Prep Work for Alignment -> Extract alpha carbon atoms of both aligned and measured residues
            1. Align reference & design PDBs
            2. Save Aligned PDB to pre-defined folder 
            3. Compute RMSD of specified residues of interest
            4. Visualize the reference PDB along with the aligned PDB
        """
        # 0. Initialize Parser and Load Structures
        parser = PDBParser()
        structure_a = parser.get_structure("A", self.ref_pdb_path)
        structure_b = parser.get_structure("B", self.design_pdb_path)

        # Assuming a simple scFv structure with a single model and chain:
        # Model 0, Chain A (usually the default single chain in AlphaFold PDBs)
        model_a = structure_a[0]
        model_b = structure_b[0]

        align_atoms_a = self.retrieve_atoms(model_a, self.ref_aligned_residues)
        align_atoms_b = self.retrieve_atoms(model_b, self.design_aligned_residues)

        measure_atoms_a = self.retrieve_atoms(model_a, self.ref_measured_residues)
        measure_atoms_b = self.retrieve_atoms(model_b, self.design_measured_residues)

        # Safety check
        if len(align_atoms_a) != len(align_atoms_b) or len(measure_atoms_a) != len(measure_atoms_b):
            print("Error: Atom lists for alignment or measurement do not match in length.")
            print(f"Alignment A: {len(align_atoms_a)}, B: {len(align_atoms_b)}")
            print(f"Measurement A: {len(measure_atoms_a)}, B: {len(measure_atoms_b)}")
            exit()
        
        # 1. Perform Superimposition (Alignment)
        super_imposer = Superimposer()
        # Superimpose Design onto Ref using the Alpha-Carbon atoms from the alignment set
        super_imposer.set_atoms(align_atoms_a, align_atoms_b)
        # Apply the transformation to the B structure
        super_imposer.apply(model_b.get_atoms())

        #2. Save Aligned PDB 
        aligned_pdb_save_path = self.save_aligned_pdb(model_b)

        #3. RMSD for Alignment Set
        # This value should be very low (typically < 1.0 Å) if the aligned residues are highly conserved.
        aligned_rmsd = super_imposer.rms
        print("-" * 50)
        print("Structural Similarity (RMSD) Analysis")
        print("-" * 50)
        print(f"RMSD of Framework Regions (FRs): {aligned_rmsd:.3f} Å")

        # Convert Atom objects to Coordinate arrays
        measured_coords_a = self.get_coords_array(measure_atoms_a)
        measured_coords_b = self.get_coords_array(measure_atoms_b) # These coordinates are now transformed!

        # Calculate RMSD
        measured_rmsd = self.compute_rmsd(measured_coords_a, measured_coords_b)

        print(f"RMSD of Measured Residues (after alignment): {measured_rmsd:.3f} Å")
        print("-" * 50)

        # 3.5 Interpretation of Results
        if measured_rmsd < 1.0:
            print("Interpretation: High structural similarity")
            print("The Aligned Residues did not significantly disrupt the measured residues' conformation.")
        elif 1.0 <= measured_rmsd < 2.0:
            print("Interpretation: Moderate difference in measured residues' conformation.")
            print("The Aligned Residues may have caused a subtle shift in measured residues interface, potentially impacting function.")
        else:
            print("Interpretation: Low structural similarity in the measured.")
            print("The Aligned Residues likely caused a major distortion of the measured residues, which could abolish function.")
        
        pymol_residue_indices = ', '.join(f"{start}-{end}" for start, end in self.design_measured_residues)

        # 4. Visualize Aligned Structures
        if visualize:
            self.visualize_aligned_proteins(file_a= self.ref_pdb_path,
                                            file_b_aligned= aligned_pdb_save_path,
                                            measured_residues_indices= pymol_residue_indices)
        
        return aligned_rmsd, measured_rmsd, aligned_pdb_save_path



