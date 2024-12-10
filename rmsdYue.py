import os
import numpy as np
from Bio.PDB import PDBParser, Superimposer


def extract_ca_atoms(structure):
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    ca_atoms.append(residue['CA'])
    return ca_atoms


def calculate_rmsd(reference, query_list):
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure('reference', reference)
    ref_atoms = extract_ca_atoms(ref_structure)

    rmsd_results = []
    for query in query_list:
        query_structure = parser.get_structure('query', query)
        query_atoms = extract_ca_atoms(query_structure)

        if len(ref_atoms) != len(query_atoms):
            print(f"Warning: Reference and Query {query} have different numbers of Cα atoms.")
            continue

        # Superimpose structures and calculate RMSD
        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, query_atoms)
        rmsd = super_imposer.rms
        rmsd_results.append((query, rmsd))

    return rmsd_results


def main():
    reference = "1avw1.pdb"  # The reference pdb file
    query_list_file = "query_list.txt"  # File containing list of query pdb files

    # Read query list from file
    with open(query_list_file, 'r') as f:
        query_list = [line.strip() for line in f.readlines()]

    # Calculate RMSD for each query compared to reference
    rmsd_results = calculate_rmsd(reference, query_list)
    if rmsd_results:
        avg_rmsd = np.mean([rmsd for _, rmsd in rmsd_results])
        print(f"Average RMSD: {avg_rmsd:.3f} Å")

	
    for query, rmsd in rmsd_results:
        print(f"RMSD between {reference} and {query}: {rmsd:.3f} Å")


if __name__ == "__main__":
    main()
