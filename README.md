# Uncertainty quantification for chemical CYP450 inhibition classification
This repo uses Monte-Carlo dropout to quantify the uncertainty and excludes the uncertain prediction against CYP450 inhibition dataset.

The datasets were parsed from PubChem BioAssay (https://pubchem.ncbi.nlm.nih.gov/bioassay/1851) and saved in ./data.

## Monte-Carlo dropout for model uncertainty quantification

Here we use Monte-Carlo dropout ("Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", arXiv preprint arXiv:1506.02142, 2015) to quantify the prediction uncertainty.

This repo uses shallow neural networks (SNNs) with various input dropout rate and 0.5 hideen layer dropout to classify whether a compound can inhibit CYP450 or not.

Each test data was predicted for 300 times and the prediction mean and standard deviation were calucalted.

<img src="https://github.com/shuan4638/MCdropout-CYP450Classifcation/blob/master/MCdropout.png">

## Excluding uncertain predictions to increase prediction accuracy

Predicted output with mean value close to 0.5 or large standard deviation were seen as uncertain prediction.

Also, one can use confidence interval (CI) to determine the uncertainty.

Excluding uncertain predictions to enhence the prediciton accuracy was demo in examaple notebook (example/example.ipynb)
<img src="https://github.com/shuan4638/MCdropout-CYP450Classifcation/blob/master/UncertainPrediction.png">

## Installation

All the chemical decription were done by rdkit. Please use conda to install rdkit.

```bash
conda install -c conda-forge rdkit
```

## Authors
This code was written by Shuan Chen (PhD candidate of KAIST CBE) in 2020 for MC dropout practice.
