# Commented out IPython magic to ensure Python compatibility.
import random, sklearn, itertools, heapq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from google.colab import files
# %matplotlib inline

########### Import dataset ###########

uploaded = files.upload()

########### Convert Dataset to a DataFrame ###########


# # Show full width and height of dataframe
pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

data = pd.read_csv("uni_test_data.csv")

data.info()

data['Churn'].value_counts()

########### Initialise ###########

debug = True
size = 53

###########  Create Initial Genome of 53 Bits ########### 

def generateGenome():
  # Total feature size is 54 - 1 (the class)
  genome = []
  for number in range(size):
      # randomly generate 0 or 1 and append to array
      genome.append(random.randint(0,1))
  return genome

############### 1. Generate Initial population ##############

# Size parameter is number of genomes in initial population
def generatePopulation(size):
    # Create a dict of population
    population = {}
    # Loop through size
    for number in range(size):
        # Append each generated genome to population dict
        population[number] = generateGenome()
    # Return population 
    if debug: print("Iniital population:", population)
    return population

# # Selected Features = X 
# X = data.drop("Churn", axis = 1)
# # Classified groups = y
# y = data["Churn"]

# # Create a training set 90% and 10% test set!
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 42 )

############### Learning Model Accuracy Value ##############

def getErrorRates(data, genome):

  features_to_drop = selectFeatures(genome)
  reduced_data = data.drop(data.columns[features_to_drop],axis=1)
  X = reduced_data.drop("Churn", axis = 1)
  y = reduced_data["Churn"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
  decision_tree = DecisionTreeClassifier(random_state=42)
  decision_tree.fit(X_train, y_train)
  predict_tree = decision_tree.predict(X_test)

  # if debug: print("Current genome has accuracy of:", accuracy_perc)
  auc = roc_auc_score(y_test, predict_tree)
  return auc

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

from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


############### Learning Model Accuracy Value ##############

def classifyValue(data, genome):

  # We want to take only the selected features chosen by the genome
  features_to_drop = selectFeatures(genome)
  # if debug: print("Features being dropped", features_to_drop)
  # Remove features from training set
  reduced_data = data.drop(data.columns[features_to_drop],axis=1)
    
  # Selected Features = X 
  X = reduced_data.drop("Churn", axis = 1)
  # Classified groups = y
  y = reduced_data["Churn"]

  # Then we split this data / large training set
  # X is the table without classification
  # y is the class
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
  
  X_train = pd.DataFrame(X_train)
  X_test = pd.DataFrame(X_test)
  y_train = pd.DataFrame(y_train)
  y_test = pd.DataFrame(y_test)

  # # Apply a standard scaling to get better optimised results
  # sc = StandardScaler()
  # X_train = sc.fit_transform(X_train_large)
  # X_test = sc.transform(X_test_small)

  # Use Decision tree classifier   
  decision_tree = DecisionTreeClassifier(random_state=42)

  decision_tree.fit(X_train, y_train)
  predict_tree = decision_tree.predict(X_test) 

  #if debug: print(classification_report(y_test_small, predict_tree))
  #if debug: print(confusion_matrix(y_test_small, predict_tree))
  accuracy_perc = accuracy_score(y_test, predict_tree)
  # if debug: print("Current genome has accuracy of:", accuracy_perc)

  # Takes the predicted items from small test size and compares it with the actual items in y_test
  # Returns the accuracy percentage of classification model =)

  # plt.clf()
  # plt.plot(fpr, tpr)
  # plt.xlabel('FPR')
  # plt.ylabel('TPR')
  # plt.title('ROC curve')
  # plt.show()
  return accuracy_perc

import random 

############### Test the learning models ##############


# This is the classification for having all 54 bits as 1's
new_array = []
for number in range(size):
      # randomly generate 0 or 1 and append to array
      new_array.append(random.randint(1,1))
if debug: print("genome:", new_array)
classifyValue(data, new_array)
print(getErrorRates(data, new_array))

import numpy as np

############### Fitness function ##############

def totalFeatures(genome):
  # This counts the total features used (i.e the 1's in the genomes)
  genome_np = np.array(genome)
  count_zero = list(genome_np)
  count_features = count_zero.count(1)
  return count_features


# Give value to genome 
def fitnessFunction(data, genome):
  # Get classification accuracy of current selected features
  error_rating = getErrorRates(data, genome)
  used_features = totalFeatures(genome)
  #if debug: print("Features used:", used_features)
  true_value = error_rating * (1-(used_features/len(genome)))
  # If errorRate is more than 50% then it is fit enough else discard 
  return true_value if (true_value >= 0.40) and (error_rating >= 0.5) else 0


# Attach a fitness value to the entire population of genomes 
def attachFitness(population):
  # First of all we calculate the value of each population
  value_pair = {"Value": [],"Features": []}

  for key, value in population.items():
    # Give fitness value to each genome (round up to 2 decimal places)
    prediction = round(fitnessFunction(data, value), 2)
    # Then we add the value AND the genome to the table
    value_pair["Value"].append(prediction)
    value_pair["Features"].append(value)

  # Then we return this dataframe table with classification accuracy value and features
  population_df = pd.DataFrame(value_pair)
  return population_df

def createDataFrame(population):
  # First of all we calculate the value of each population
  value_pair = {"Accuracy": [], "True Value": [],"Total Features": [], "Features": [], "AUC": []}

  for key, value in population.items():
    # Give fitness value to each genome (round up to 2 decimal places)
    prediction = round(fitnessFunction(data, value), 2)
    accuracy = classifyValue(data, value)
    total_features = totalFeatures(value)
    error = getErrorRates(data, value)

    # Then we add the value AND the genome to the table
    value_pair["True Value"].append(prediction)
    value_pair["Accuracy"].append(accuracy)
    value_pair["Total Features"].append(total_features)
    value_pair["Features"].append(value)
    value_pair["AUC"].append(error)

  # Then we return this dataframe table with classification accuracy value and features
  dataframe_graph = pd.DataFrame(value_pair)
  dataframe_graph = dataframe_graph.sort_values(["True Value"], ascending=False)
  dataframe_graph = dataframe_graph.reset_index(drop=True)
  return dataframe_graph

############### Selection of the Fittest ##############


def tournament(population):
  # We want to take 2 random genomes to battle it out
  winners = {}
  total_winners = 0
  while total_winners < 2:
    # Pick genome 1 at random
    genome_one = random.randint(0, len(population)-1)
    # Pick genome 2 at random
    genome_two = random.randint(0, len(population)-1)

    # Get the fitness of each genome from the total weight (if it's over limit then its 0)
    first_genome = list(population.items())[genome_one][1]
    second_genome = list(population.items())[genome_two][1]

    # Calculate classification accuracy of each feature selection
    fitness_one = fitnessFunction(data, first_genome)
    fitness_two = fitnessFunction(data, second_genome)   
    # Fittest genome gets added to the winners pot!
    if fitness_one > fitness_two and fitness_one > 0:
      winners[total_winners] = population[genome_one]
      total_winners += 1
      if debug: print("Tournament", total_winners, ": Genome one wins")
    elif fitness_two > fitness_one and fitness_two > 0:
      winners[total_winners] = population[genome_two]
      total_winners += 1
      if debug: print("Tournament", total_winners, ": Genome two wins")
  # Returns 2 winners as a dict
  if debug: print("New Parents:", winners)
  return winners

############### Select the Elitest ##############


def elitism(dataframe):
  # Find the top 20% of population (round up)
  ten_percent = round(dataframe.shape[0] * 0.2)
  # Sort in order from largest to smallest accuracy of classification
  dataframe = dataframe.sort_values(["Value"], ascending=False)
  # Top 10% percentage
  top_ten = dataframe.head(ten_percent).reset_index()
  dataframe = dataframe.reset_index(drop=True)
  # Append all of the items in "Features" into a dict
  best_dict = top_ten["Features"].to_dict();
  if debug: print("Elitest:", best_dict)
  return best_dict

############### Crossover Function ##############

def crossover(parents):
    # Split parents out
    parent_one = parents[0]
    parent_two = parents[1]
    # Randomly selects an int between 2n and the length of genome 
    # I've selected items-1 as counting starts from 0 and we want total length of features
    pointer = random.randint(2,len(parent_one)-1)
    # Takes second half of genome 1 
    p1_tmp = parent_one[pointer:]
    # Takes second half of genome 2
    p2_tmp = parent_two[pointer:]
    # Takes first half of genome 1
    p1_tmp2 = parent_one[:pointer]
    # Takes first half of genome 2
    p2_tmp2 = parent_two[:pointer]
    # Makes a new genome
    p1_tmp2.extend(p2_tmp)
    p2_tmp2.extend(p1_tmp)
    # Append 2 new genomes to results
    result = []
    result.append(p1_tmp2)
    result.append(p2_tmp2)
    # Returns 2 child genomes as a new solution! 
    return result

############## Mutation Function ##############

# We want to 'mutate' aka flip a bit 
# If a problem uses 10 bits, then a good default mutation rate would be (1/10) = 0.10 or a probability of 10 percent.

def mutation(children):
    # Mutation rate using 1/53(features) = 0.01851851851 = 0.02
    # Did not vary the genome so putting it back to 0.0
    mutation_rate = 0.02
    # Iterate through each bit
    for i in range(len(children)):
        for j in range(len(children[i])):
            # Generate a random number between 0 to 1 for bit 
            bit_rate = random.uniform(0,1)
            # print(mutation_rate, bit_rate)
            # If bit rate is less than mutation rate then we flip the bit
            if mutation_rate > bit_rate:
                if children[i][j] == 1:
                    children[i][j] = 0
                    # print("0", children[i][j])
                else:
                    children[i][j] = 1
                    # print("1", children[i][j]) 
    if debug: print("Crossover & Mutated Children:", children)                 
    return children

############### Create Population ##############

# This function creates the population by going through the tourney etc etc process until
# ... the full number of population has been reached 
# Population parameter = starting population 
# Size parameter = number of genomes in a population

def createPopulation(population, size):

  # A dictionary containing all of new population  
  new_population = {}
  # Current population we are creating 
  count = 0

  # Evaluate value for each item in population and create a DataFrame (only used to find the Elite)
  evaluate_population = attachFitness(population)
  elite = elitism(evaluate_population)

  # Append the most elite to the dictionary first
  new_population = elite
  # Start the count from lenth of elitism list (i.e if there are 3 starts from 3)
  count += len(new_population)

  while count < size:
    parents = tournament(population)
    children = crossover(parents)
    mutated = mutation(children)
    # Add the new children to the dict by getting the value of the 1st & 2nd item in dict
    new_population[count] = mutated[0]
    new_population[count] = mutated[1]
    count += 1
  if debug: print("New population:", new_population)
  return new_population

############### Testing Station ##############

def geneticAlgorithm(data, size, generations):
    
    # Create initial population
    next_population = generatePopulation(size)

    # Initialise generation counter
    counter = 0
    found_optima = False

    # Create a pandas dataframe add the initial population to it 
    all_population = pd.DataFrame(next_population, columns=['Features','Total Features','Accuracy','True Value','AUC'])

    while counter < generations:
      global mean_dict
      # create population from previous population
      next_population = createPopulation(next_population, size)
      if debug: print("Generation",counter + 1,":", next_population)
        # Append current generation to a panda dataframe
        # all_populations = all_populations.append(attachFitness(next_population), ignore_index=True)

      # Converts current population into a dataframe
      create_dataframe = createDataFrame(next_population)
      print("New Population Created!:", create_dataframe)

      # Calculate the Average for each item
      mean_dict["Average Accuracy"].append(create_dataframe["Accuracy"].mean())
      mean_dict["Average Features"].append(create_dataframe["Total Features"].mean())
      mean_dict["Average Value"].append(create_dataframe["True Value"].mean())
      mean_dict["Average AUC"].append(create_dataframe["AUC"].mean())
      print(mean_dict)

      # Add winning genome to population
      all_population = all_population.append(create_dataframe, ignore_index=True)
      # Apend the average to another dataframe      all_population = all_population.append(create_dataframe, ignore_index=True)
      counter += 1

    print("All genomes made!", all_population)
    return all_population

############### Testing Station ##############

pd.set_option('display.max_rows', None)

# Size paramter is number populations in a generation
# Generation parameter is number of generations of populations
# Data parameter is the dataset used

# Create a dict
mean_dict = {"Average Accuracy": [], "Average Features": [], "Average Value": [], "Average AUC": []}

# Run the GA scripts!
df_ga = geneticAlgorithm(data, 20, 80)

# Convert dict to dataframe 
mean_df = pd.DataFrame(mean_dict, columns=["Average Accuracy", "Average Features", "Average Value", "Average AUC"])
mean_df

# Export to CSV
mean_df.to_csv('higher-auc-pop20gen80-undersample-full-mean.csv') 
files.download('higher-auc-pop20gen80-undersample-full-mean.csv')



# Calculate actual learning model accuracy for each genome
new_df = df_ga.copy()
new_df

# Export to CSV
new_df.to_csv('pop20gen80-undersample-full.csv') 
files.download('pop20gen80-undersample-full.csv')

classifyValue(data, [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0])