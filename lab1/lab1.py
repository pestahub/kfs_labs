#! /usr/bin/python3

import numpy as npy
from numpy import linalg as lnlg
import matplotlib.pyplot as mpl

data_read = npy.genfromtxt('testLab1Var40.csv', delimiter=',')
time = data_read[:, 0]
time = time[:, npy.newaxis]
current = data_read[:, 1]
current = current[:, npy.newaxis]
voltage = data_read[:, 2]
voltage = voltage[:, npy.newaxis]
print(voltage[10], time[10])
fig, (ay1, ay2) = mpl.subplots(2, 1, sharex=True)
T_per = 0.1
ay1.plot(time[time < 2*T_per], voltage[time < 2*T_per])
ay1.grid()
ay1.set_xlabel('time, s')
ay1.set_ylabel('voltage, V')
ay2.plot(time[time < 2*T_per], current[time < 2*T_per])
ay2.grid()
ay2.set_xlabel('time, s')
ay2.set_ylabel('current, A')
mpl.show()
fig.savefig('Recieved data(part)')
X = npy.concatenate([voltage[0:len(voltage) -
                             2], current[0:len(current)-2]], axis=1)
Y = current[1:len(current)-1]
K = npy.dot(npy.dot(lnlg.inv(npy.dot(X.T, X)), X.T), Y)
Td = 0.001
R = 1 / K[0] * (1 - K[1])
T = -Td / npy.log(K[1])
L = T*R
print('R = ', R, 'T = ', T, 'L = ', L)
current_est = X.dot(K)
fig, ax = mpl.subplots(1, 1)
mpl.plot(time[time < T_per], current[time < T_per])
mpl.plot(time[time < T_per], current_est[time[0:len(current)-2] < T_per])
ax.grid()
ax.set_xlabel('time, s')
ax.set_ylabel('current, A')
mpl.show()
fig.savefig('Compared data(part)')
R_est = []
L_est = []

n = 1000
for i in range(0, n-1, 1):
    ind = (time >= T_per*i) & (time <= T_per*(i+1))
    new_current = current[ind]
    new_current = new_current[:, npy.newaxis]
    new_voltage = voltage[ind]
    new_voltage = new_voltage[:, npy.newaxis]
    X = npy.concatenate([new_voltage[1:len(new_voltage) - 1],
                        new_current[0:len(new_current) - 2]], axis=1)
    Y = current[1:len(new_current) - 1]
    K = npy.dot(npy.dot(lnlg.inv(npy.dot(X.T, X)), X.T), Y)
    if K[1] > 0:
        R = 1/K[0]*(1-K[1])
        T = -Td / npy.log(K[1])
        R_est.append(R)
        L_est.append(T*R)
R_est = npy.array(R_est)
L_est = npy.array(L_est)
print('Mean value of R: ', npy.mean(R_est), ' Ohm')
print('Standart deviation of R: ', npy.std(R_est))
print('Mean value of L = ', npy.mean(L_est), ' Hn')
print('Standart deviation of R: ', npy.std(L_est))
