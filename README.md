# misregulation-simulations
To simulate the evolutionary dynamics of transcription factor binding sites from the manuscript 
"Gene misregulation can help alleviate maladaptation" by A. Wagner

The repository contains three files
misregulation_wagner_popsim_main.py: Python script for the core of the simulation program

popfuncs.c: several functions used in main to implement mutation, selection, and data collection

mutation_data.txt: data file loaded into main which contains the likelihoods that a mutation
creates or destroys transcription factor binding sites (TFBSs), pased on mouse PBM data 

