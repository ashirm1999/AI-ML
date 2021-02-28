from Gates import Model
import numpy as np


# AND GATE
training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))
labels = np.array([1,0,0,0])
p_object = Model(2)
p_object.training(training_inputs, labels)
test_input = np.array([1,1])

# OR GATE
training_inputs_o = []
training_inputs_o.append(np.array([1,1]))
training_inputs_o.append(np.array([1,0]))
training_inputs_o.append(np.array([0,1]))
training_inputs_o.append(np.array([0,0]))
labels_o = np.array([1,1,1,0])
p_object_o = Model(2)
p_object_o.training(training_inputs_o, labels_o)
test_input_o = np.array([0,0])


print("************************************************************")
print("AND GATE")
print("Input to model = ", test_input, end=' , ')
print("Output of trained model = ", p_object.prediction(test_input))
print("************************************************************")
print("OR GATE")
print("Input to model = ", test_input_o, end=' , ')
print("Output of trained model = ", p_object_o.prediction(test_input_o))
print("************************************************************")