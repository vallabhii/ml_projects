import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def compute_cost_fxn(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):
        f_wb=np.dot(w,x[i])+b
        cost+=(f_wb-y[i])**2
    cost=cost/(2*m)
    return(cost)

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=np.zeros_like(w)
    dj_db=0.0
    for i in range(m):
        f_wb=np.dot(w,x[i])+b
        dj_dw+=(f_wb-y[i])*x[i]
        dj_db+=f_wb-y[i]
    dj_dw=dj_dw/m
    dj_db=dj_db/m     
    return dj_dw,dj_db

def gradient_descent(x,y,cost_function,gradient_function,alpha,num_iterations):
    m=x.shape[0]
    w_in=np.zeros(x.shape[1])
    b_in=0
    J_history=[]
    w_history=[]
    for i in range(num_iterations):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        w=w-alpha*dj_dw               
        b=b-alpha*dj_db 
        if i<num_iterations: 
            cost=cost_function(x,y,w,b)
            J_history.append(cost)
        step = max(1, num_iterations // 10)
        if i % step == 0:
            w_history.append(w.copy())
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")     
    return w,b,J_history,w_history

