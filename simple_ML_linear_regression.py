import numpy as np

from simple_ML_model import MY_NET_linear_regression

np.random.seed(123)

def generate_x_and_y(w_num = np.random.randint(5, 12), size = np.random.randint(1000, 2000)):
    # print(w_num, batch_size)
    w_num = w_num
    size = size
    X = np.array(np.random.randn(size, w_num))
    X_TEST = np.array(np.random.randn(50, w_num), dtype = float) #test数据共50个
    real_w = np.random.randn(w_num) * 10
    noise = np.random.randn(size) * 1
    noise_test = np.random.randn(50) * 1
    Y = real_w.dot(X.T) + noise
    Y_TEST = real_w.dot(X_TEST.T) + noise_test
    return X, Y, X_TEST, Y_TEST

X, Y, X_TEST, Y_TEST = generate_x_and_y()
w_num = X.shape[1]
batch_size = 20
eta = 0.1
model = MY_NET_linear_regression(w_num = w_num, batch_size = batch_size, eta = eta)

loss = model.train(X, Y, epoch = 2)
acc, acc_array = model.predict(X_TEST, Y_TEST)
print("acc = ", acc, "acc_array = ", acc_array)
