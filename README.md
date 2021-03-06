# avianInfluenza
This is a project of predicting the cross-species transmission risk of avian influenza virus isolates.
The sequences of influenza A H7N9 virus strains with each containing 8 complete segments were downloaded from Influenza Research DataBase (IRD) and gisaid database, and their non-coding region at both ends were deleted, and then the 8 segments of each strain were concatenated into a genomic sequence. Each genomic sequence was given a label "0" or "1" if it is isolated from avian or human,respectively. The genomic sequences were aligned and each nucleotide position was represented by a numeric code (0,1,2,3,4 for A, T, G, C, gap, respectively). Models were trains with different weights for avian and human classes. Accuracy was calculted for avian and human classes with 5-fold cross-validation. A model which achieve 100% accuracy for human class and highest accuary for avian class is chosen as the final model for using the sequences of the 8 segments of a avian influenza strain as the input to prediction the human infectivity of this strain. A server for this purpose is available at http://124.16.144.116:8099/h7n9/Index/home.do. 

Reference

Computational Predicting the Human Infectivity of H7N9 Influenza Viruses Isolated from Avian Hosts. 2020. Transboundary and Emerging Diseases. 
DOI: 10.1111/tbed.13750
