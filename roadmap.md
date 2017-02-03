## Problem
Determine if pairs of Quora questions are duplicate
[https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs]
400k pairs, 37% positive

## Preprocessing
* Spell-check with Google search

## Feature extraction
* BOW Tokenizations
* Sum pretrained GloVe vectors

## Evaluation
* F1

## Methods to try
* smart undersampling in the majority class by choosing hard examples in mini-batch allocation [http://www.idiap.ch/~fleuret/SMLD/2014/SMLD2014_-_Olivier_Canevet_-_Efficient_mining_of_hard_examples_for_object_detection.pdf]
