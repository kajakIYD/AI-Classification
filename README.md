# AI-Classification
TensorFlow and scikit-learn sandbox for classification purposes. 

## Sick or well heart disease
A little bit less famous classification task, that helps to figure out who suffers from heart disease based on several numerical attributes. Techniques used: **PCA**, **SGDClassifier**, **SVC**, **RandomForest**, **DecisionTree**
Very useful tool in context of analysis and comparison of different is **confusion matrix**. These are listed below:

![SGD](https://github.com/kajakIYD/AI-Classification/blob/master/SickOrWellHeartDisease/DocumentationImages/SGD_full_set.PNG)

![SVC](https://github.com/kajakIYD/AI-Classification/blob/master/SickOrWellHeartDisease/DocumentationImages/SVC_full_set.PNG)

![DecisionTree](https://github.com/kajakIYD/AI-Classification/blob/master/SickOrWellHeartDisease/DocumentationImages/DecisionTree_full_set.PNG)

**Dimensionality reducion** was made, explained variance versus dimensions plot looks like below:

![Explained_variance_vs_dimensions](https://github.com/kajakIYD/AI-Classification/blob/master/SickOrWellHeartDisease/DocumentationImages/Explained_variance_vs_dimensions.PNG)

So reducing dimensionality into three dimensions explains near 100% of dataset variance, so newly-reduced dataset would have only three features. However, now it impossible to tell what exactly these features represent.

Precision ranking: [{'name': 'SGDClassifier(a', 'value': 0.8571428571428571}, {'name': 'SVC(C=0.01, cac', 'value': 0.8571428571428571}, {'name': 'RandomForestCla', 'value': 0.8}, {'name': 'DecisionTreeCla', 'value': 0.8}, {'name': 'DecisionTreeCla reduced', 'value': 0.6857142857142857}, {'name': 'DecisionTreeCla reduced', 'value': 0.6571428571428571}, {'name': 'DecisionTreeCla reduced', 'value': 0.6285714285714286}, {'name': 'DecisionTreeCla reduced', 'value': 0.45714285714285713}]

Recall ranking: [{'name': 'RandomForestCla', 'value': 0.9032258064516129}, {'name': 'SVC(C=0.01, cac', 'value': 0.8823529411764706}, {'name': 'SGDClassifier(a', 'value': 0.8571428571428571}, {'name': 'DecisionTreeCla reduced', 'value': 0.8}, {'name': 'DecisionTreeCla reduced', 'value': 0.7857142857142857}, {'name': 'DecisionTreeCla', 'value': 0.7777777777777778}, {'name': 'DecisionTreeCla reduced', 'value': 0.6052631578947368}, {'name': 'DecisionTreeCla reduced', 'value': 0.47058823529411764}]

Accuracy ranking: [{'name': 'SVC(C=0.01, cac', 'value': 0.881578947368421}, {'name': 'SGDClassifier(a', 'value': 0.868421052631579}, {'name': 'RandomForestCla', 'value': 0.868421052631579}, {'name': 'DecisionTreeCla', 'value': 0.8026315789473685}, {'name': 'DecisionTreeCla reduced', 'value': 0.7763157894736842}, {'name': 'DecisionTreeCla reduced', 'value': 0.75}, {'name': 'DecisionTreeCla reduced', 'value': 0.6447368421052632}, {'name': 'DecisionTreeCla reduced', 'value': 0.5131578947368421}]

**Best accuracy: {'name': 'SVC(C=0.01, cac', 'value': 0.881578947368421} <br>Best precision: {'name': 'SGDClassifier(a', 'value': 0.8571428571428571} <br>Best recall: {'name': 'RandomForestCla', 'value': 0.9032258064516129}**

## Clothes Classification
Famous MNIST classification task. Done with usage of **TensorFlow** and **Keras**

## Flower Classification
Famous iris classification task. Done with usage of **TensorFlow** and **Keras**
