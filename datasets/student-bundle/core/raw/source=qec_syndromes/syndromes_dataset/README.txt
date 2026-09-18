The file names: d-<surface_code_distance>_pfr-<physical_fault_rate>_nb-<number_of_samples>

The file format is a csv file with the following columns:
- label: binary label (0: no error, 1: error)
- syndromes: syndrome measurement sequence (tuples of the form (round, syndromes))
- quantity: number of samples for this label + syndrome sequence