Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

Model Details

This project uses a Random Forest Classifier to predict whether an individual's income exceeds $50K per year. The model is trained on processed census data with categorical features encoded using OneHotEncoder and the label binarized.

Intended Use

The model is intended for educational purposes and for demonstrating how to build and deploy a scalable machine learning pipeline. It can be used to predict high-income individuals in datasets with similar features, but should not be used for real-world financial decision-making without further validation.

Training Data

The training data consists of processed records from the U.S. census dataset. Categorical features include workclass, education, marital status, occupation, relationship, race, sex, and native country. The label is a binary indicator of income above or below $50K per year.

Evaluation Data

The evaluation data is a held-out test split from the same census dataset. Performance metrics were computed on the entire test set and on slices of the data for each categorical feature to examine model behavior across subgroups.

Metrics

The model achieves the following overall performance on the test set: precision of 0.7419, recall of 0.6384, and F1 score of 0.6863. For slices of the data, performance varies across different feature values. For example, for the workclass feature, precision ranges from 0.6538 to 1.0, recall ranges from 0.4048 to 1.0, and F1 ranges from 0.5 to 1.0. Similar variations are observed across education, marital status, occupation, relationship, race, sex, and native country slices.

Ethical Considerations

This model may reflect biases present in the training data, such as disparities across race, sex, or socioeconomic status. Predictions should not be used for decisions that could negatively affect individuals based on these attributes. Transparency about potential biases is critical when interpreting results.

Caveats and Recommendations

This model is a demonstration of a machine learning pipeline and is not suitable for high-stakes decisions. Users should carefully consider ethical implications and limitations. Further evaluation, additional data, and bias mitigation strategies are recommended before any deployment in real-world scenarios.
