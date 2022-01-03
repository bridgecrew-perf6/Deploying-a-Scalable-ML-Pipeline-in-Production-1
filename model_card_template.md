# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Felipe F Melo created the model. It is Decision Tree Classifier using the default hyperparameters in scikit-learn 1.0.

## Intended Use
This model should be used to predict the salary based on some personal features

## Metrics
The model was evaluated using Precision, Recall and fbeta. For one of the iteration of the for loop, fbeta = 0.649, precision = 0.710, recall = 0.599.

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The data is split in 80% of data for training phase and 20% for test phase.



## Ethical Considerations
Data related to race and gender.

## Caveats and Recommendations
Improve the model and change more the parametes.