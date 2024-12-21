# Experiment Analysis

## Download Annotated Dataset

Experimental analysis requires human clinician annotations, which were generated separately (the above pipeline only produces the unannotated dataset and labels generated by the VeriFact AI system).

```sh
# Download Dataset with Human Annotations from Physionet
# Replace verifact with this repository's root path
# Replace ${PHYSIONET_USERNAME} with your physionet username
wget -r -N -c -np --directory-prefix=data --user ${PHYSIONET_USERNAME} --ask-password https://physionet.org/files/verifact-bhc/1.0/
# Downloaded dataset is at verifact/data/physionet.org/files/verifact-bhc/1.0
```

## Analysis Scripts

The following scripts generate experimental results:

1. `1_proposition_validity.py`: Analysis of how well each proposition conforms to definition of logical proposition.
2. `2_interrater_between_human_clinicians.py`: Agreement between initial three clinician that annotated each proposition.
3. `3_verifact_vs_ground_truth_tables.py`: Agreement between each hyperparameter variant of VeriFact AI System versus the human clinican ground truth labels.
4. `4_verifact_vs_ground_truth_label_distributions.py`: Analysis of the label distribution and agreement of one of the best VeriFact AI hyperparameters versus the human clinican ground truth labels.
5. `5_verifact_vs_ground_truth_sensitivity_analysis.py`: Sensitivity analysis of four hyperparameters in the VeriFact AI system and its effect on performance.
6. `6_verifact_weak_to_strong_trajectory.py`: Analysis of how label assignments change when transitioning from a weak VeriFact AI system to a strong VeriFact AI system by sweeping the number of retrieved facts.