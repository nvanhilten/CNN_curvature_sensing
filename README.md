All scripts and data used for CNN training on MD data studying curvature sensing by helical peptides.

When using this code, please cite:
Van Hilten, N.; Methorst, J.; Verwei, N.; Risselada, H.J., Science Advances. 2023, 9(11). DOI: 10.1126/sciadv.ade8839

This repo contains:
- data.txt			The original data set with sequences in first column and MD-calculated ddF values in second column
- random_test_set.txt		The randomly generated test set containing used for the hyperparameter testing and the final model evalutions
- sequences.txt			An example sequence file for lipid packing defect sensing prediction by predict.py

- best_model.h5			The final trained CNN model that was used for all the data presented in the paper

- Run_CNN.py			Python script for CNN training (Usage: python3 Run_CNN.py -f <input data>)
- General_functions.py		Functions used by Run_CNN.py
- predict.py			Python script for CNN prediction for an input sequence file (Usage: python3 predict.py -s <sequence file> -m <trained model>)
