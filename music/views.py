from django.shortcuts import render
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator


def page1(request):
      return render(request, "page1.html")

def getNeighbors(trainingSet, instance, k):
      distances = []
      for x in range(len(trainingSet)):
            dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
            distances.append((trainingSet[x][2], dist))

      distances.sort(key=operator.itemgetter(1))
      neighbors = []
      for x in range(k):
            neighbors.append(distances[x][0])

      return neighbors

# identify the class of the instance
def nearestClass(neighbors):
      classVote = {}

      for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVote:
                  classVote[response] += 1
            else:
                  classVote[response] = 1

      sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

      return sorter[0][0]

# function to evaluate the model
def getAccuracy(testSet, prediction):
      correct = 0
      for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                  correct += 1

      return (1.0 * correct) / len(testSet)

def distance(instance1 , instance2 , k ):
      distance =0 
      mm1 = instance1[0] 
      cm1 = instance1[1]
      mm2 = instance2[0]
      cm2 = instance2[1]
      distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
      distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
      distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
      distance-= k
      return distance

def detect(request):
      # directory that holds the wav files
      directory = "./genres/"

      # binary file where we will collect all the features extracted using mfcc (Mel Frequency Cepstral Coefficients)
      f = open("my.dat", 'wb')

      i = 0

      for folder in os.listdir(directory):
            i += 1
            if i == 11:
                  break
            for file in os.listdir(directory+folder):        
                  try:
                        (rate, sig) = wav.read(directory+folder+"/"+file)
                        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                        covariance = np.cov(np.matrix.transpose(mfcc_feat))
                        mean_matrix = mfcc_feat.mean(0)
                        feature = (mean_matrix, covariance, i)
                        pickle.dump(feature, f)
                  except Exception as e:
                        print('Got an exception: ', e, ' in folder: ', folder, ' filename: ', file)        

      f.close()

      # Split the dataset into training and testing sets respectively
      dataset = []

      def loadDataset(filename, split, trSet, teSet):
            with open('my.dat', 'rb') as f:
                  while True:
                        try:
                              dataset.append(pickle.load(f))
                        except EOFError:
                              f.close()
                              break
            for x in range(len(dataset)):
                  if random.random() < split:
                        trSet.append(dataset[x])
                  else:
                        teSet.append(dataset[x])
      trainingSet = []
      testSet = []
      loadDataset('my.dat', 0.66, trainingSet, testSet)

      # making predictions using KNN
      leng = len(testSet)
      global predictions
      predictions = []
      for x in range(leng):
            predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

      accuracy1 = getAccuracy(testSet, predictions)
      print(accuracy1)

      if request.method=="POST":
            uploaded_file=request.POST['filepath']

      test_file = uploaded_file

      (rate, sig) = wav.read(test_file)
      mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
      covariance = np.cov(np.matrix.transpose(mfcc_feat))
      mean_matrix = mfcc_feat.mean(0)
      feature = (mean_matrix, covariance, i)

      from collections import defaultdict
      results = defaultdict(int)

      directory = "./genres/"

      i = 1
      for folder in os.listdir(directory):
            results[i] = folder
            i += 1

      pred = nearestClass(getNeighbors(dataset, feature, 5))
      print(results[pred])
      rs=results[pred]

      content = {
      'rs':"The music is identified to be:",
      'result':rs,
      }
      return render(request, "page1.html",content)  