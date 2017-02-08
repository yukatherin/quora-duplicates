## Mentor
?

## Problem Description
Determine if pairs of Quora questions are duplicate. It is interesting to aggregate answers on duplicate questions, and suggest similar questions to users.

## Data
[https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs]
400k pairs, 37% positive

## Methodology/Algorithm
* LSTM with attention
* smart undersampling in the majority class by choosing hard examples in mini-batch allocation [http://www.idiap.ch/~fleuret/SMLD/2014/SMLD2014_-_Olivier_Canevet_-_Efficient_mining_of_hard_examples_for_object_detection.pdf]

## Related Work
Papers about LSTM with attention:
(to determine logical entailment) Rocktäschel et al. 100D LSTMs w/ word-by-word attention. [https://arxiv.org/pdf/1509.06664v1.pdf]

## Evaluation Plan
* F1
