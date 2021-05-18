import numpy as np


class MY_NET_linear_regression:
    #第一步初始化参数，w_num是w的个数（w1，w2，...）， batch_size是每个batch的大小，eta是学习速率
    def __init__(self, w_num, batch_size, eta):
        self.w_num = w_num
        self.batch_size = batch_size
        self.w_array = np.random.randn(1, w_num) #这里将w_array看作行向量
        self.b = np.random.randn(1)
        self.eta = eta

    def __fit(self, X, Y, i): #注意这里Y是行向量 ， 注意这里输入的X,Y是全局的X,Y
        x_train_batch = X[i * self.batch_size : (i + 1) * self.batch_size]
        y_train_batch = Y[i * self.batch_size : (i + 1) * self.batch_size]
        Z = self.w_array.dot(x_train_batch.T) + self.b
        loss = (Z - y_train_batch).dot((Z - y_train_batch).T)  #Z - y_train_batch  是一行， Z - y_train_batch的转置是一列
        dw, db = self.__backprop(x_train_batch, y_train_batch, Z, self.batch_size)#反向传播计算dw，db
        return loss, dw, db

    #定义反向传播函数
    def __backprop(self, X, Y, Z, batch_size):
        dw = list()
        #Z = [w1*x11+w2*x12+...+wn*x1n , w1*x21+...+wn*x2n , ... , w1*xn1+...+wn*xnn]，假设loss对w1求偏导，得到的是[x11, x21, ... , xn1]，自己验证
        for i in range(self.w_num):
            d_wi = 2 * (Z - Y) * X[: , i : i + 1].T
            # print(type(d_wi), d_wi.shape, batch_size, X[: , i : i + 1].T.shape)
            d_wi_average = np.sum(d_wi) / batch_size
            dw.append(d_wi_average)
        
        tem_db = 2 * (Z - Y)
        db = np.sum(tem_db) / batch_size
        dw = np.array(dw)
        return dw, db
    
    def __update(self, dw, db):
        self.w_array = self.w_array - self.eta * dw
        self.b = self.b - self.eta * db

    def predict(self, X, Y):
        prediction = self.w_array.dot(X.T)
        acc_array = abs(1 - abs((Y - prediction)/ Y))
        acc = np.sum(acc_array, axis = 1) / acc_array.shape[1]
        # print(abs(1 - abs((Y - prediction)/ Y))[0 : 20])
        # print(prediction[0 : 20])
        return acc, acc_array
        

    def train(self, X, Y, epoch): #这里的X,Y是全局的X,Y
        total_loss = dict()
        #由于X.shape[0] 不一定整除 self.batch_size， 所以不能直接用X.shape[0] 除以 self.batch_size
        round = (X.shape[0] // self.batch_size) + 1
        for epo in range(epoch):    
            loss = list()
            for i in range(round - 1):
                L, dw, db = self.__fit(X, Y, i)
                self.__update(dw, db)
                loss.append(L)

            #由于X.shape[0] 不一定整除 self.batch_size，所以要把最后一个round拿出来单独处理
            #这样做的好处是省去了判断i是否等于round - 1 的步骤，但是却多了这么多代码
            x_last_batch = X[(round - 1) * self.batch_size :]
            y_last_batch = Y[(round - 1) * self.batch_size :]
            last_batch_size = X.shape[0] - (round - 1) * self.batch_size
            Z = self.w_array.dot(x_last_batch.T) + self.b
            L = (Z - y_last_batch).dot((Z - y_last_batch).T)
            dw, db = self.__backprop(x_last_batch, y_last_batch, Z, last_batch_size)
            self.__update(dw, db)
            loss.append(L)
            total_loss[str(epo)] = loss
        
        return total_loss


class MY_NET_perceptron:
    def __init__(self, eta) -> None:
        self.w_num = 2
        self.batch_size = 4
        self.eta = eta
        self.w_array = np.random.rand(self.w_num) * 10
        self.b = np.random.rand(1)

    def _forward_prop(self, X, Y):
        l = 0
        for i in range(len(X)):
            if (np.dot(self.w_array, X[i]) + self.b) * Y[i] <= 0:
                l = l - (np.dot(self.w_array, X[i]) + self.b) * Y[i]

        return l

    def _backprop(self, X, Y):
        dw = 0
        db = 0
        for i  in range(len(X)):
            if (np.dot(self.w_array, X[i]) + self.b) * Y[i] <= 0:  #记得加上self.b啊！！找了一个小时终于找到了
                dw -= X[i] * Y[i]
                db -= np.sum(Y[i])   #！！！刚开始写成了db -= np.sum(Y)，找了快两个小时啊
                # print('dw = ', dw, 'db = ', db, 'i = ', i)
        return dw, db

    def _update(self, dw, db):
            self.w_array -= self.eta * dw
            self.b -= self.eta * db        

    def train(self, epoch, X, Y):
        loss = []
        for epo in range(epoch):
            l = self._forward_prop(X, Y)
            dw, db = self._backprop(X, Y)
            self._update(dw, db)
            if epo % 50 == 0:
                loss.append(l)
        return loss
 
    def predict(self, X):
        predict_Y = []
        for x in X:
            if np.dot(self.w_array, x) + self.b > 0:
                predict_Y.append(1)
            else:
                predict_Y.append(0)
        return predict_Y
