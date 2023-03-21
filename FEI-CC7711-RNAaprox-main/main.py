import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from statistics import mean, stdev

print('Carregando Arquivo de teste')
arquivo = np.load('teste2.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])


def simulacao():
    iter = 10000

    regr = MLPRegressor(hidden_layer_sizes=(2),
                        max_iter=iter,
                        activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=5000)
    print('Treinando RNA')
    regr = regr.fit(x,y)



    print('Preditor')
    y_est = regr.predict(x)

    print(regr.loss_)






    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)




    plt.show()
    return regr.loss_
e = []
i = 0
while i < 10:
    e.append(simulacao())
    i+=1

print(mean(e))
print(stdev(e))