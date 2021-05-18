import copy

import numpy as np

from simple_ML_model import MY_NET_perceptron


def train_data(data: str):
    if data == 'and':
        X = np.array(([1, 0], [1, 1], [0, 1], [0, 0]))
        Y = np.array([-1, 1, -1, -1])
    elif data == 'or':
        X = np.array(([1, 0], [1, 1], [0, 1], [0, 0]))
        Y = np.array([1, 1, 1, -1])

    return X, Y

class model_gate(MY_NET_perceptron):
    def __init__(self, eta, data_mode:str) -> None:
        self.data_mode = data_mode
        self.X, self.Y = train_data(self.data_mode)
        super().__init__(eta)


eta = 0.005
epoch = 1000
#or网络
my_or_model = model_gate(eta = eta, data_mode = 'or')
loss_or = my_or_model.train(epoch = epoch, X = my_or_model.X, Y = my_or_model.Y)
predict_Y_or = my_or_model.predict(X = my_or_model.X)
print(predict_Y_or)
print(loss_or)
#and网络
my_and_model = model_gate(eta = eta, data_mode = 'and')
loss = my_and_model.train(epoch = epoch, X = my_and_model.X, Y = my_and_model.Y)
predict_Y = my_and_model.predict(X = my_and_model.X)
print(predict_Y)
print(loss)

#xor网络,通过两个与门和一个或门构造
and_gate1 = copy.deepcopy(my_and_model)
and_gate2 = copy.deepcopy(my_and_model)
or_gate = copy.deepcopy(my_or_model)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = []
for input in inputs:
    A, B = input
    A_inv = 1 - A
    B_inv = 1 - B
    and_gate1_input = [[A, B_inv]]# 第一个与门的输入是[A, B反]
    and_gate2_input = [[A_inv, B]]# 第二个与门的输入是[A反, B]，注意格式是[[]]，由于predict函数实现的问题，只能这样了
    and_gate1_output = and_gate1.predict(X = and_gate1_input)
    and_gate2_output = and_gate2.predict(X = and_gate2_input)
    or_gate_output = or_gate.predict(X = [[and_gate1_output, and_gate2_output]]) #将两个与门的输出输入到或门中
    output.append(or_gate_output.pop()) #将或门的输出存下来，用pop是因为或门的输出格式是list，且这个list只有单个元素，具体细节见predict

for i in range(len(inputs)):
    print('input : ', inputs[i], 'output : ', output[i])
