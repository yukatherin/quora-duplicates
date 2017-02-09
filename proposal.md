# CS224N Project Proposal

## Team Members
Katherine Yu (yukather), Genki Kondo (genki), Ramon Tuason (rtuason)

## Mentor
Kai Sheng Tai

## Problem Description
Determine if pairs of Quora questions containing similar words are exact duplicates in meaning.  The solution to this problem is interesting to aggregate answers on duplicate questions, and suggest similar questions to users posting new questions on Quora.

## Data
[https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs]
400k pairs, 37% positive.
The pairs are hand-labeled. Examples:
* Example 1: 
  * Question1 - "How can I increase the speed of my internet connection while using a VPN?" 
  * Question2 - "How can Internet speed be increased by hacking through DNS?"
  * This pair is negative since hacking through DNS is to increase internet speed is different from increasing speed while you are using VPN.
* Example 2:
  * Question1 - "Why are so many Quora users posting questions that are readily answered on Google?"
  * Question2 - "Why do people ask Quora questions which can be answered easily by Google?"
  * This pair is positive since the meanings are exactly equivalent up to human judgement: Quora users posting questions is the same as people asking Quora questions, and "readily" is equivalent to "easily."


## Methodology/Algorithm
* Baseline: 
  * Simple 1-layer methods like logistic regression or SVM will not work well without feature engineering relatedness-features (e.g. cosine similarity on tf-idf), which we don't want time on. 
  * We think our baseline for question relatedness should be a Siamese net, with shared parameters W and b,  h1=f(Wx1+b), h2=f(Wx2+b) where x1 and x2 are the summed GloVe vectors for questions 1 and 2, an activation layer between h1 and h2, and cross entropy loss on the output of the activation layer.
* Primary goals:
   * Baseline: Siamese net with summed GloVe vectors
   * LSTM with attention
* Secondary goals:
   * smart undersampling in the majority class by choosing hard examples in mini-batch allocation [http://www.idiap.ch/~fleuret/SMLD/2014/SMLD2014_-_Olivier_Canevet_-_Efficient_mining_of_hard_examples_for_object_detection.pdf]
   * character learning/two-byte encodings

## Related Work
Basic paper about LSTM with attention (to determine logical entailment): 
* Rocktäschel et al. 100D LSTMs w/ word-by-word attention. [https://arxiv.org/pdf/1509.06664v1.pdf]

## Evaluation Plan
* F1
* We will produce tuning plots for hyperparameters: 
    * alpha in cross entropy loss to weight positive examples versus negative examples
* We will use cross-validation with F1, look at confusion matrix, error analysis by hand.
