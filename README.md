# Heart-Disease-Classifier-
This code will classify different factors affecting the heart diseases using Decision Tree Classifier Algorhithm.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
heart=pd.read_csv("heart.csv")
heart
heart.isnull()
heart.head()
heart.describe()
heart.loc[(heart['age'].isnull() | heart['sex'].isnull() | heart['chol'].isnull() | heart['trestbps'].isnull())]
# We can extract the data in this format from pandas like this:
all_inputs = heart[['sex', 'age',
 'chol', 'trestbps']].values
# Similarly, we can extract the class labels
all_labels = heart['age'].values
from sklearn.model_selection import train_test_split
(training_inputs,
testing_inputs,
training_classes,
testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25,random_state=1)
from sklearn.tree import DecisionTreeClassifier
# Create the classifier
decision_tree_classifier = DecisionTreeClassifier()
# Train the classifier on the training set
decision_tree_classifier.fit(training_inputs, training_classes)
# Validate the classifier on the testing set using classification accuracy
decision_tree_classifier.score(testing_inputs, testing_classes)
model_accuracies = []
for repetition in range(1000):
 (training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_labels, test_size=
0.25)

 decision_tree_classifier = DecisionTreeClassifier()
 decision_tree_classifier.fit(training_inputs, training_classes)
 classifier_accuracy = decision_tree_classifier.score(testing_inputs, testing_classes)
 model_accuracies.append(classifier_accuracy)

plt.hist(model_accuracies)
;
import numpy as np
from sklearn.model_selection import StratifiedKFold
def plot_cv(cv, features, labels):
 masks = []
 for train, test in cv.split(features, labels):
  mask = np.zeros(len(labels), dtype=bool)
 mask[test] = 1
 masks.append(mask)

 plt.figure(figsize=(15, 15))
 plt.imshow(masks, interpolation='none', cmap='gray_r')
 plt.ylabel('Fold')
 plt.xlabel('Row #')
plot_cv(StratifiedKFold(n_splits=10), all_inputs, all_labels)
from sklearn.model_selection import cross_val_score
decision_tree_classifier = DecisionTreeClassifier()
# cross_val_score returns a list of the scores, which we can visualize
# to get a reasonable estimate of our classifier's performance
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_labels
, cv=10)
plt.hist(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))
decision_tree_classifier = DecisionTreeClassifier(max_depth=1)
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_labels
, cv=10)
plt.hist(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))
