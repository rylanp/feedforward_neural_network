import csv
from random import random as rand
from random import uniform
from random import shuffle as shuffle_list
import math
const_maxActivation = 9
def Activation(x):
	return const_maxActivation / (1 + (math.exp(-x)))
	# value = x
	# if (x < 0):
	# 	value = 0
	# return value
def ReLUActivationDerivative(weightedInput):
	activation = Activation(weightedInput)
	return activation * (1 - (activation / const_maxActivation))
	# value = 1
	# if (weightedInput < 0):
	# 	value = 0
	# return value
def NodeCost(outputActivation, expectedOutput):
	error = (outputActivation - expectedOutput)
	return error * error
def NodeCostDerivative(outputActivation, expectedOutput):
	return 2 * (outputActivation - expectedOutput)
class Neuron:
	def __init__(self, index, n_inputs):
		self.index = index
		self.bias = 0 # random between -1 and 1
		self.value = 0
		self.inputWeights = []
		for n in range(0, n_inputs):
			value = ((rand() * 2) - 1) / math.sqrt(n_inputs)
			self.inputWeights.append(value)
		self.weightedInputValue = 0
		self.inputvalues = []
		self.inputActivations = []
		#print(f"Weight = {self.inputWeights}")
	def __str__(self):
		return f"(Neuron{self.index} bias: {self.bias} value: {self.value})"
class NeuronGradient:
	def __init__(self, index, n_inputs):
		self.index = index
		self.biasGradient = 0
		self.inputWeightGradients = []
		for n in range(0, n_inputs):
			self.inputWeightGradients.append(0)
	def __str__(self):
		return f"(Neuron{self.index} bias: {self.bias} value: {self.value})"
class DataPoint:
	def __init__(self, inputValue, expectedoutput):
		self.inputValue = inputValue
		self.expectedoutput = expectedoutput
		self.team = ""
class Network:
	def __init__(self, inputs, n_hidden_layers, n_neurons, n_outputs):
		self.inputs = inputs
		self.n_hidden_layers = n_hidden_layers
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self.hidden_layers = []
		self.gradients_respective = []
		self.initWeights_Biases()
		self.initGradients()
		self.recentCost = 0.8

		# for the learning process
		#TODO: keep track of cost and weight gradients
	def initWeights_Biases(self):
		for layer in range(0, self.n_hidden_layers):
			l = []
			for index in range(0, self.n_neurons):
				l.append(Neuron(index=index, n_inputs=(self.inputs if layer == 0 else self.n_neurons)))
			self.hidden_layers.append(l)

		l = []
		for outputIndex in range(0, self.n_outputs):
			l.append(Neuron(index=outputIndex, n_inputs=self.n_neurons))
		self.hidden_layers.append(l)
	def initGradients(self):
		for layer in range(0, self.n_hidden_layers):
			l = []
			for index in range(0, self.n_neurons):
				l.append(NeuronGradient(index=index, n_inputs=(self.inputs if layer == 0 else self.n_neurons)))
			self.gradients_respective.append(l)

		l = []
		for outputIndex in range(0, self.n_outputs):
			l.append(NeuronGradient(index=outputIndex, n_inputs=self.n_neurons))
		self.gradients_respective.append(l)
	def forward(self, X):
		for layerIndex in range(0, len(self.hidden_layers)):
			for neuronIndex in range(0, len(self.hidden_layers[layerIndex])):
				sum = 0
				self.hidden_layers[layerIndex][neuronIndex].inputvalues = []
				self.hidden_layers[layerIndex][neuronIndex].inputActivations = []
				weights = self.hidden_layers[layerIndex][neuronIndex].inputWeights
				for weightIndex in range(0, len(weights)):
					if (layerIndex == 0):
						sum += weights[weightIndex] * X[weightIndex]
						self.hidden_layers[layerIndex][neuronIndex].inputvalues.append(weights[weightIndex] * X[weightIndex])
						self.hidden_layers[layerIndex][neuronIndex].inputActivations.append(X[weightIndex])
					else:
						sum += self.hidden_layers[layerIndex -1][weightIndex].value * weights[weightIndex]
						self.hidden_layers[layerIndex][neuronIndex].inputvalues.append(self.hidden_layers[layerIndex -1][weightIndex].value * weights[weightIndex])
						self.hidden_layers[layerIndex][neuronIndex].inputActivations.append(self.hidden_layers[layerIndex -1][weightIndex].value)
				# get value of node in previous layer
				self.hidden_layers[layerIndex][neuronIndex].weightedInputValue = self.hidden_layers[layerIndex][neuronIndex].bias + sum
				self.hidden_layers[layerIndex][neuronIndex].value = Activation(self.hidden_layers[layerIndex][neuronIndex].bias + sum)
		outputs = []
		outputNodesIndex = len(self.hidden_layers) - 1
		for i in range(0, len(self.hidden_layers[outputNodesIndex])):
			value = self.hidden_layers[outputNodesIndex][i].value
			outputs.append(value)
		return outputs
	def train(self, trainingData, learnRate, portion, limit_iterations):
		iterations = []
		costs = []

		while (len(iterations) < limit_iterations):
			self.learnWithDerivatives(trainingData, learnRate, portion)
			avgcost = self.recentCost
			print(f"Iteration: {len(iterations)} | {avgcost}")

			iterations.append(len(iterations))
			costs.append(avgcost)
		entries = len(trainingData)
		correct = 0

		averageCost = network.averageCost(trainingData)
		print(f"Cost: {averageCost}")
		outputs = []

		for point in trainingData:
			out = network.forward(point.inputValue)

			out.append(point.inputValue[0])
			out.append(point.inputValue[1])
			out.append(point.expectedoutput[0])
			out.append(point.expectedoutput[1])
			out.append(point.team)
			outputs.append(out)
		for output in outputs:
			string = "Incorrect"
			value = 0
			if (round(output[0]) == round(output[4])):
				value += 0.5
				correct += 0.5
				string = "Close"
			if (round(output[1]) == round(output[5])):
				value += 0.5
				correct += 0.5
				string = "Close"
			if (value >= 1):
				string = "Correct"
			print(f"{output[6]} ({output[2]} {output[3]}) | gives {round(output[0],1)} - {round(output[1],1)} | expected {output[4]} - {output[5]} | {string}")
		print(f"Percentage: {correct} / {entries} = {math.ceil((correct / entries) * 100)}%")	
	def learnWithDerivatives(self, trainingData, learnRate, portion):
		shuffled_data = trainingData
		shuffle_list(shuffled_data)
		portion_training_data = []
		num = math.ceil((portion * len(trainingData)))
		
		for n in range(0, num):
			data = shuffled_data[n]
			portion_training_data.append(data)

		for dataPoint in portion_training_data:
			self.updateAllGradients(dataPoint)
		self.applyGradients(learnRate / len(portion_training_data))
		self.clearAllGradients()
		#remove in the actual thing, but this gives me an update
		self.recentCost = self.averageCost(trainingData)
	def cost(self, dataPoint):
		outputs = self.forward(dataPoint.inputValue)
		cost = 0
		for outputIndex in range(0, len(outputs)):
			cost += NodeCost(outputs[outputIndex], dataPoint.expectedoutput[outputIndex])
		return cost
	def averageCost(self, dataPoints):
		totalCost = 0
		for point in dataPoints:
			totalCost += float(self.cost(point))
		return totalCost / len(dataPoints)
	def applyGradients(self, learnRate):
		#go through every weight and bias
		#take that respective 
		for layerIndex in range(0, len(self.hidden_layers)):
			for neuronIndex in range(0, len(self.hidden_layers[layerIndex])):

				gradient_bias = self.gradients_respective[layerIndex][neuronIndex].biasGradient
				self.hidden_layers[layerIndex][neuronIndex].bias -= gradient_bias * learnRate

				gradient_weights = self.gradients_respective[layerIndex][neuronIndex].inputWeightGradients
				weights = self.hidden_layers[layerIndex][neuronIndex].inputWeights
				for weightIndex in range(0, len(weights)):
					self.hidden_layers[layerIndex][neuronIndex].inputWeights[weightIndex] -= gradient_weights[weightIndex] * learnRate
	def updateAllGradients(self, dataPoint):
		outputs = self.forward(dataPoint.inputValue)
		# for the output layer
		output_layer = self.hidden_layers[len(self.hidden_layers) - 1]
		nodeValues = self.calculateOutputLayerNodeValues(output_layer, dataPoint.expectedoutput)
		self.updateOutputLayerGradients(len(self.hidden_layers) - 1, nodeValues)

		# wroking on
		for hiddenLayerIndex in range((len(self.hidden_layers) - 2), -1, -1):
			#should go through last hidden layer first, then work way backwards
			nodeValues = self.calculateHiddenLayerNodeValues(hiddenLayerIndex, nodeValues)
			self.updateHiddenLayerGradients(hiddenLayerIndex, nodeValues)
	def updateOutputLayerGradients(self, layer_index, nodeValues):
		# we want: weight += previous layer activation + nodevalue
		# we want: bias += 1 * nodeValues
		current_layer = self.hidden_layers[layer_index]
		prev_layer = self.hidden_layers[layer_index - 1]
		for neuronIndex in range(0, len(current_layer)):
			for weightedInputIndex in range(0, len(current_layer[neuronIndex].inputWeights)):
				derivativeCostWeight = current_layer[neuronIndex].inputActivations[weightedInputIndex] * nodeValues[neuronIndex]
				self.gradients_respective[layer_index][neuronIndex].inputWeightGradients[weightedInputIndex] += derivativeCostWeight

			derivativeCostBias = 1 * nodeValues[neuronIndex]
			self.gradients_respective[layer_index][neuronIndex].biasGradient += derivativeCostBias
	def calculateOutputLayerNodeValues(self, layer, outputs):
		nodeValues = []
		for i in range(0, len(outputs)):
			# output node activation value, then the expected value
			costDerivative = NodeCostDerivative(layer[i].value, outputs[i]) # correct
			# x is the respective weight input
			activationDerivative = ReLUActivationDerivative(layer[i].weightedInputValue) # correct
			nodeValues.append(activationDerivative * costDerivative)
		return nodeValues
	def calculateHiddenLayerNodeValues(self, layer_index, old_node_values):
		new_node_values = [] # length = number of nodes out for this layer

		last_layer = self.hidden_layers[layer_index + 1]
		current_layer = self.hidden_layers[layer_index]

		for neuron_index in range(0, len(current_layer)):
			weightedConnection = 0
			for node_value_index in range(0, len(old_node_values)):

				weightedConnection += old_node_values[node_value_index] * last_layer[node_value_index].inputWeights[neuron_index]
			new_node_values.append(weightedConnection * ReLUActivationDerivative(current_layer[neuron_index].weightedInputValue))

		return new_node_values
	def updateHiddenLayerGradients(self, layer_index, nodeValues):
		current_layer = self.hidden_layers[layer_index]
		for neuronIndex in range(0, len(current_layer)):
			for weightedInputIndex in range(0, len(current_layer[neuronIndex].inputWeights)):
				derivativeCostWeight = current_layer[neuronIndex].inputActivations[weightedInputIndex] * nodeValues[neuronIndex]
				self.gradients_respective[layer_index][neuronIndex].inputWeightGradients[weightedInputIndex] += derivativeCostWeight

			derivativeCostBias = 1 * nodeValues[neuronIndex]
			self.gradients_respective[layer_index][neuronIndex].biasGradient += derivativeCostBias
	def clearAllGradients(self):
		# resets all gradients to zero
		for layer in range(0, len(self.hidden_layers)):
			for neuron_index in range(0, len(self.hidden_layers[layer])):
				self.gradients_respective[layer][neuron_index].biasGradient = 0
				length = len(self.gradients_respective[layer][neuron_index].inputWeightGradients)
				self.gradients_respective[layer][neuron_index].inputWeightGradients = [0]*length
	def saveValues(self, network_size):
		with open("EPL_network_biases_weights.csv", 'w', newline='') as f:
			thewriter = csv.writer(f)
			thewriter.writerow(['inputs','hidden','neurons','outputs'])
			thewriter.writerow(network_size)
			for layer in self.hidden_layers:
				for neuron in layer:
					thewriter.writerow(neuron.inputWeights)
					# write to file
	def loadValues(self, network_size):
		with open("EPL_network_biases_weights.csv") as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			lines = list(readCSV)
			lines.pop(0)
			info = lines.pop(0)
			same = True
			for index in range(0, len(info)):
				if (int(info[index]) != network_size[index]):
					same = False
					print("ERROR: Not the same size")
					break
			if (same):
				index = 0
				layer = 0
				neuron = 0
				weight_index = 0
				for row in lines:
					for weight in row:
						self.hidden_layers[layer][neuron].inputWeights[weight_index] = float(weight)
						weight_index += 1
					neuron += 1
					weight_index = 0
					if (neuron == len(self.hidden_layers[layer])):
						neuron = 0
						layer += 1
def getInputs(inputs):
	l = []
	# [t1point, t2points, score1, score2, t1-t2 stats]
	team1stats = teams[inputs[0]]
	team2stats = teams[inputs[1]]
	for stat_index in range(0,len(team1stats)):
		l.append(team1stats[stat_index] - team2stats[stat_index])
	#point = DataPoint(l, [0,0])
	return l
		

teams = {}
with open("epl_teams_data.csv") as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	lines = list(readCSV)
	lines.pop(0)
	for row in lines:
		l = []
		for i in range(1,len(row)):
			l.append(float(row[i]))
		teams[row[0]] = l
X = []
with open("epl_scores_data.csv") as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	lines = list(readCSV)
	lines.pop(0)
	for row in lines:
		l = []
		# [t1point, t2points, score1, score2, t1-t2 stats]
		team1stats = teams[row[0]]
		team2stats = teams[row[1]]
		for stat_index in range(0,len(team1stats)):
			l.append(team1stats[stat_index] - team2stats[stat_index])
		point = DataPoint(l, [int(row[2]),int(row[3])])
		point.team = f"{row[0]} vs {row[1]}"
		X.append(point)

# train network with these inputs
hidden_layers = 3
neurons = 8
outputs = 2
network_size = [len(X[0].inputValue), hidden_layers, neurons, outputs]
network = Network(len(X[0].inputValue), hidden_layers, neurons, outputs)
network.loadValues(network_size)
network.train(X, 0.0001, 0.3, 3000)
network.saveValues(network_size)

round_value = 0
games = [['AVL','SOU'],
				['NFO','FUL'],
				['BHA','CRY'],
				['WOL','MCI'],
				['NEW','BOU'],
				['TOT','LEI'],
				['BRE','ARS'],
				['ARS','BRE'],
				['EVE','WHU'],
				['MUN','CHE'],
				['LIV','LEE']]
for game in games: 
	inputs = getInputs(game)
	outputs = network.forward(inputs)
	g1 = round(outputs[0], round_value)
	g2 = round(outputs[1], round_value)
	if (round_value == 0):
		g1 = int(g1)
		g2 = int(g2)
	print(f"{game[0]}  {game[1]}  {g1} -  {g2}")
# cmd B to build

'''
feed in----
CHE,7,LEI,1
t1, t1 points, t2, t2 points
expect(t1 goals, t2 goals)
'''