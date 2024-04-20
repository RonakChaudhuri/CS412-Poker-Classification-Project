import numpy as np
from sklearn import svm

#because the data is not properly set, I have to completely rework it into seperate arrays
def reprocess(data):
  X = np.array([])
  Y = np.array([])
  counter = 0
  
  for a in data:
    
    check = a.find(",")
    newData = a
    newArray = np.array([])
    
    while check != -1:
      newArray = np.append(newArray, newData[:check])
      newData = newData[check+1:]
      check = newData.find(",")
      
    if counter == 0:
      X = newArray
      Y = np.array([newData])
    else:
      Y = np.concatenate((Y , newData), axis = None)
      X = np.vstack((X, newArray))
    counter += 1
    
    
  return X, Y

#Testing Data To Use
poker_test = np.loadtxt("poker-hand-testing.data", dtype='str')
testx, testy = reprocess(poker_test)

#Training Data To Use
poker_train = np.loadtxt("poker-hand-training-true.data", dtype='str')
trainx, trainy = reprocess(poker_train)

clf = svm.SVC(kernel="linear", C=1000)
clf.fit(trainx, trainy)

svm_predictions = clf.predict(testx)
#print(svm_predictions)
