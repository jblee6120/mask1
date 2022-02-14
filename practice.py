import numpy as np
import scipy.special
class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    
        self.inodes = inputnodes #입력 레이어의 노드 수 설정
        self.hnodes = hiddennodes #히든 레이어의 노드 수 설정
        self.onodes = outputnodes #출력 레이어의 노드 수 설정

        self.lr = learningrate #학습률 설정
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list): #신경망 훈련
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0-final_outputs)), np.transpose(hidden_outputs)) #학습률과 오차를 반영하여 가중치를 수정하는 공식을 코드로 구현

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))



        pass

    def query(self, inputs_list): #신경망에 질문

        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.001

#신경망 객체 만들기
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

input = [0.7, 0.4, 0.2]
target = [0.01, 0.7, 0.5]
epochs = 10000000

for i in range(epochs+1):
  n.train(input, target)
  if i == epochs:
    total_error = (target - n.query(input)) **2
    print('epochs:{}'.format(i))
    print('total error')
    print(total_error)
    print('current value')
    print(n.query(input))