import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path='D:/2345Downloads/练习2/ex2data2.txt'
data=pd.read_csv(path,header=None,names=['Test1','Test2','Accepted'])
data.head()
positive2=data[data['Accepted'].isin([1])]
negative2=data[data['Accepted'].isin([0])]
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test1'],positive2['Test2'],s=50,c='b',marker='o',label='Accepted')
ax.scatter(negative2['Test1'],negative2['Test2'],s=50,c='r',marker='x',label='Rejected')
ax.legend()
ax.set_xlabel=('Test1 Scores')
ax.set_ylabel=('Test2 Scores')
plt.show()
degree=6
data2 = data
x1=data2['Test1']
x2=data2['Test2']
data2.insert(3,'Ones',1)
# 把序号为3的列后面加入一列名为Ones，值全为1的数据
for i in range(1,degree):
    for j in range(0,i+1):
        data2['F'+str(i-j)+str(j)]=np.power(x1,i-j)*np.power(x2,j)
# x1的i-j次方和x2的j次方，增加特征量
data2.drop('Test1',axis=1,inplace=True)
data2.drop('Test2',axis=1,inplace=True)
# 删除数据集中的某一列，inplace为True表示直接在数据集data2上修改，输出data2为修改后的状态；inplace为False，表示新建一个数据集修改内容
# data2仍然为原状，data3为修改后的集合。drop,删除行、删除列,默认行:axis = 1,删除列,axis=0删除行
# 将x1，x2两个特征值增加到了6个
data.head()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta,X,y,alpha):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg=(alpha/(2*len(X)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/len(X)+reg
def gradientReg(theta, X, y, alpha):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    error=sigmoid(X*theta.T)-y
    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        if i==0:
            grad[i]=np.sum(term)/len(X)
        else:
            grad[i]=(np.sum(term)/len(X))+((alpha / len(X)) * theta[:,i])
    return grad
col=data2.shape[1]
X2=data2.iloc[:,1:col]
y2=data2.iloc[:,0:1]
theta2=np.zeros(col-1)
X2=np.array(X2.values)
y2=np.array(y2.values)
alpha=1
cost(theta2,X2,y2,alpha)
import scipy.optimize as opt
result2 = opt.fmin_tnc(func=cost, x0=theta2, fprime=gradientReg, args=(X2, y2, alpha))
result2
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
def hfunc2(theta,x1,x2):
    temp=theta[0][0]
    place=0
    for i in range(1,degree):
        for j in range(0,i+1):
            temp+=np.power(x1,i-j)*np.power(x2,j)*theta[0][place+1]
            place+=1
    return temp   #这里的temp是算的整个边界值，temp是累加和，x1和x2的数据按循环7次后相加，得到temp

def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)
# 将-1到1.5之间的数按等间距分为1000个数，即起点为-1终点为1.5的等差数列，个数为1000
    cordinates=[(x,y) for x in t1 for y in t2]  #x,y都变成1000的平方个数
    x_cord, y_cord = zip(*cordinates)    #此处是将x和y元素分开
# zip函数是将a = [1,2,3]，b = [4,5,6]，c = [4,5,6,7,8]
#     zipped = zip(a,b)=[(1, 4), (2, 5), (3, 6)]     # 打包为元组的列表
#     zip(a,c)=[(1, 4), (2, 5), (3, 6)]   # 元素个数与最短的列表一致
# zip（*）将合并好的列表解压，变为原来的元素
# zip(*zipped)=# [(1, 2, 3), (4, 5, 6)]          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
    h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])  #增加hval一列，转化成28维，即终值temp来划定边界范围
    decision = h_val[np.abs(h_val['hval']) < 2* 10 ** -3]   #小于这个数就是边界值，边界值就是调参？
    return decision.x1, decision.x2
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test1'], positive2['Test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test1'], negative2['Test2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test1 Score')
ax.set_ylabel('Test2 Score')
x, y = find_decision_boundary(result2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()
