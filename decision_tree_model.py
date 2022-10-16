# Commented out IPython magic to ensure Python compatibility.
import random, sklearn, itertools, heapq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from google.colab import files
# %matplotlib inline

########### Import dataset ###########

uploaded = files.upload()

########### Second dataset ###########


re_uploaded = files.upload()



########### Convert Dataset to a DataFrame ###########

debug = True

# # Show full width and height of dataframe
pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

#data = pd.read_csv("full_data_cleaned_numerical(not downsampled).csv")
data = pd.read_csv("uni_data_downsampled_numerical.csv")
data

data["Churn"].value_counts()

############### Select Features from Genome ##############


# Returns fitness value for each genome
# First we want to take the bit 
# Then we use this iterate through and select only features with 1 

def selectFeatures(genome):
  drop_columns = []
  # This selects only the features that are bit = 0 (not selected)
  for key, bit in enumerate(genome):
    # Checks if 1 or 0
    if bit == 0:
      # Add to column number to array
      drop_columns.append(key)
  # Returns an array of all the column indexes to drop!
  return drop_columns

# full_genome = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, , 1, 1, 1,1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, , 1, 1, 1, , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

genome_7 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
genome = [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

############### Learning Model  ##############

# We want to take only the selected features chosen by the genome
features_to_drop = selectFeatures(genome)
# if debug: print("Features being dropped", features_to_drop)
# Remove features from training set
model_data = data.drop(data.columns[features_to_drop],axis=1)
  
# Selected Features
#input_data = model_data

#ALL features 
input_data = data.copy()

# Selected Features = X 
X = input_data.drop("Churn", axis = 1)
# Classified groups = y
y = pd.DataFrame(input_data["Churn"])

# Then we split this data / large training set
# X is the table without classification
# y is the class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Apply a standard scaling to get better optimised results
# sc = StandardScaler()
# # Use Decision tree classifier 

# X_train = sc.fit_transform(X_train)

# # # Apply to test as well but ONLY to transform (not fit)
# X_test = sc.transform(X_test)

hyper_param = {
'ccp_alpha': 0.0,
 'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 1}],
 'criterion': 'gini',
 'max_depth': None,
 'max_features': None,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'random_state': 42,
 'splitter': 'best'}


decision_tree = DecisionTreeClassifier(criterion="gini", max_depth = 21, random_state = 42)

# criterion="gini", class_weight = {0:0.5, 1: 0.5}, random_state = 42

y_score = decision_tree.fit(X_train, y_train).predict(X_test) 


##### Print 

if debug: print(classification_report(y_test, y_score))
if debug: print(confusion_matrix(y_test, y_score))

accuracy_perc = accuracy_score(y_test, y_score)
if debug: print("Accuracy:", accuracy_perc)


auc = roc_auc_score(y_test, y_score)
if debug:  print('AUC score: %.3f' % auc)

tree_depth =  decision_tree.tree_.max_depth
if debug:  print('Tree depth:', tree_depth)

########### Plot Confusion Matrix ###########


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(decision_tree, X_test, y_test)
plt.show()



########### Prune da tree! ###########


path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

# Train decision tree using the effective alphas. 
# The last value in ccp_alphas is the alpha value that prunes the whole tree, leaving the tree, clfs[-1], with one node.

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(decision_tree)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [decision_tree.tree_.node_count for decision_tree in clfs]
depth = [decision_tree.tree_.max_depth for decision_tree in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [decision_tree.score(X_train, y_train) for clf in clfs]
test_scores = [decision_tree.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

y_true = y_test

from sklearn import metrics

# calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
fpr
tpr

import matplotlib.pyplot as plt

######### Draw ROC/AUC Curve  ######### 

plt.rcParams["figure.figsize"] = (20,10)


plt.plot(fpr, tpr,linewidth=1, markersize=1, color='blue')

plt.xlabel("Generations")
plt.ylabel("Average Accuracy")

############### Evaluation  ##############

from sklearn import metrics, model_selection

metrics.plot_roc_curve(decision_tree, X_test, y_test)

# Calculate the AUC 

metrics.auc(fpr, tpr)

# Visualise Decision Tree

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
feature_cols = X_test.columns
export_graphviz(decision_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

