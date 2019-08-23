# For NN
import torch
import torch.optim as optim

# For Graph
import numpy as np
import matplotlib.pyplot as plt

# For preparing data
import csv

## Prepare Data
data = []
c = 0
with open("./data/breastcancer_training.csv") as csvfile:
	reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
	for row in reader: # each row is a list
		if c > 0:
			data.append(row)
		c = c + 1

target = []
for row in data:
	target.append([int(row.pop())])

#for x in range(len(data)):
#	print(data[x])
#	print(target[x])
#	print("")

######



#### Custom Network Module
class ANN(torch.nn.Module):
	def __init__(self):
		super(ANN, self).__init__()
		
		self.fc1 = torch.nn.Linear(9, 12)
		self.fc2 = torch.nn.Linear(12, 1)
		
	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		
		return x

# Load Model from local "my_model.py"
load_model = False
		
# DATASET
X = torch.tensor((data), dtype=torch.float)
y = torch.tensor((target), dtype=torch.float)

#### LOAD FROM TRAINING
if load_model == True:
	nn = torch.load("./my_model.py")
	nn.eval()
#### TRAINING
else: 
	# START
	nn = ANN()

	optimizer = optim.SGD(nn.parameters(), lr=0.1) # Stochastic Gradient Descent
	criterion = torch.nn.MSELoss() # Criterion to Measure the mean squared error 

	epochs = 500 # Count of Epochs
	
	score = [] # For printing in Graph
	
	for e in range(epochs):
		running_loss = 0
		for x in range(0, len(data)):
			out = nn(X[x]) # 

			optimizer.zero_grad() # Set Gradient to Zero
			
			loss = criterion(out, y[x]) # Calculate Loss
			
			loss.backward() # Backpropagation
			
			optimizer.step() # Update Weight
			running_loss += loss.item()
			params = list(nn.parameters())
		else:
			score.append(running_loss)
			print("Epoch: "+str(e)+" running_loss: "+str(running_loss)) # Print out actual Epoch
####

### PREPARE VALIDATION DATA

## Prepare Data
data = []
c = 0
with open("./data/breastcancer_test.csv") as csvfile:
	reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
	for row in reader: # each row is a list
		if c > 0:
			data.append(row)
		c = c + 1

target = []
for row in data:
	target.append([int(row.pop())])

X = torch.tensor((data), dtype=torch.float)
y = torch.tensor((target), dtype=torch.float)
	
# Print Out Result
print("Result:")		
sum_wrong = 0
#sum_wrong = 0
sum = 0
for _x, _y in zip(X, y):
	prediction = nn(_x)
	print('Input:\t', list(map(int, _x)))
	print('Pred:\t', int(prediction >= 0.5))
	print('Ouput:\t', int(_y))
	
	if int(prediction >= 0.5) != int(_y):
		sum_wrong = sum_wrong + 1
	
	sum = sum + 1
	
	print('######')
	
print("Summe: "+str(sum))
print("Wrong: "+str(sum_wrong))
print("Accuracy: "+str((sum_wrong / sum) * 100))
#print(sum / sum_wrong)
	
if load_model == False:
	# Print Graph
	plt.plot(score)
	plt.axis([0, epochs, 0, np.amax(score)])
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	plt.show()
	
# Save Model
torch.save(nn, "./my_model.py")
