# To read csv files
import pandas as pd

trainingData = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter="\t", quoting=3)
testData = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

unlabeledTrainingData = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# We verify the number of reviews read in total.

print "Successfully read %d labeledTrainData, %d testData"\
"and %d unlabeledTrainingData\n" %(trainingData["review"].size, testData["review"].size, unlabeledTrainingData["review"].size)
