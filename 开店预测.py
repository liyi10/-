import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# a=np.eye(5)
# print(a)
path='D:/2345Downloads/ex1data1.txt'
data=pd.read_csv(path,header=None,names=['population','profit'])
data.head()
data.plot(kind='scatter',x='population',y='profit',figsize=(12,8))
def computerCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
data.insert(0, 'Ones', 1)
cols = data.shape[1]#表示输出表格的列数，实际为3列
X = data.iloc[:,:-1]#X是data里的除最后列,前者是表示输出行，后者表示列，
                    # 且从第一列输出到倒数第二列，-1表示最后一列，但是步长默认为1
                    #输出停止在倒数第二行
y = data.iloc[:,cols-1:cols]#y是data最后一列，即是2：3，但是列的序号是从0开始的
                            #按序号来说是到号2输出就结束了
X.head()#head()是观察前5行
y.head()
X = np.matrix(X.values)#将X转变为矩阵
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))#theta值为0
X.shape, theta.shape, y.shape
computerCost(X,y,theta)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))#这里形成了theta行列数的0矩阵
    parameters = int(theta.ravel().shape[1])#将theta的列数输出
    cost = np.zeros(iters)#输出翼iters数目的0矩阵，行向量，iters列

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computerCost(X, y, theta)

    return theta, cost
alpha = 0.01
iters = 1500
g, cost = gradientDescent(X, y, theta, alpha, iters)
g
predict1 = [1,3.5]*g.T
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)
#预测35000和70000城市规模的小吃摊利润
x = np.linspace(data.population.min(), data.population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)#这里的【0，0】是指矩阵g的第0行（序号）第0列，第0行第1列
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.population, data.profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
#原始数据以及拟合的直线
