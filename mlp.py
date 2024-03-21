#part 2

from scipy.special import expit # sigmoid function
import numpy as np
x1=np.array([0,0,0,0,1,1,1,1])
x2=np.array([0,0,1,1,0,0,1,1])
x3=np.array([0,1,0,1,0,1,0,1])
 
#out - truth value
functions=["x1 ^ x2 ^ x3","x1 v x2 v x3", "(x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) ∧ x3"," "]
func=0
#    x1 ^ x2 ^ x3
y1=np.array([0,0,0,0,0,0,0,1])

#    x1 v x2 v x3
y2=np.array([0,1,1,1,1,1,1,1]) 

# (x1 ∧ ¬x2) ∨ (¬x1 ∧ x2) ∧ x3
y3=np.array([0,0,0,1,0,1,0,0]) 
 
logicalfn= [y1,y2,y3]

np.random.seed(30)
# for y_1
w11_t = np.random.normal(-1,1)
w12_t = np.random.normal(-1,1)
w13_t = np.random.normal(-1,1)
w10_t = np.random.normal(-1,1)

# for y_2
w21_t = np.random.normal(-1,1)
w22_t = np.random.normal(-1,1)
w23_t = np.random.normal(-1,1)
w20_t = np.random.normal(-1,1)

w1_t = np.random.normal(-1,1)
w2_t = np.random.normal(-1,1)
w0_t = np.random.normal(-1,1)

epoch = 10000
lr = 0.05

MSE = np.zeros([epoch,1])
for y in logicalfn:
    for i in range(epoch):
        #forward pass
        
        #output layer
        y_1 =expit(w11_t*x1 + w12_t*x2 + w13_t*x3 + w10_t)
        y_2 =expit(w21_t*x1 + w22_t*x2 + w23_t*x3 + w20_t)
        y_h=expit(w2_t*y_2 + w1_t*y_1 + w0_t)
    
        #bacward pass
    
        #loss layer
        LMSE_losslayer= 2*(y-y_h)*-1
    
        #output layer
        LMSE_outputlayer_y1 = LMSE_losslayer*(1-y_h)*(y_h)*w1_t
        LMSE_outputlayer_y2 = LMSE_losslayer*(1-y_h)*(y_h)*w2_t
    
        #Update weights
    
        #output layer
        w1_t = w1_t - lr * np.sum(LMSE_losslayer*(1-y_h)*(y_h)*y_1)
        w2_t = w2_t - lr * np.sum(LMSE_losslayer*(1-y_h)*(y_h)*y_2)
        w0_t = w0_t - lr * np.sum(LMSE_losslayer*(1-y_h)*(y_h)*1)
    
        #hidden layer
        w11_t = w11_t - lr * np.sum(LMSE_outputlayer_y1*(1-y_1)*y_1*x1)
        w12_t = w12_t - lr * np.sum(LMSE_outputlayer_y1*(1-y_1)*y_1*x2)
        w13_t = w13_t - lr * np.sum(LMSE_outputlayer_y1*(1-y_1)*y_1*x3)
        w10_t = w10_t - lr * np.sum(LMSE_outputlayer_y1*(1-y_1)*y_1*1)
    
        w21_t = w21_t - lr * np.sum(LMSE_outputlayer_y2*(1-y_2)*y_2*x1)
        w22_t = w22_t - lr * np.sum(LMSE_outputlayer_y2*(1-y_2)*y_2*x2)
        w23_t = w23_t - lr * np.sum(LMSE_outputlayer_y2*(1-y_2)*y_2*x3)
        w20_t = w20_t - lr * np.sum(LMSE_outputlayer_y2*(1-y_2)*y_2*1)
    
    y_1 =expit(w11_t*x1 + w12_t*x2 + w13_t*x3 + w10_t)
    y_2 =expit(w21_t*x1 + w22_t*x2 + w23_t*x3 + w20_t)
    y_h=expit(w2_t*y_2 + w1_t*y_1 + w0_t)
    print("\nGiven Funtion:",functions[func])
    func+=1
    #print(y_1)
    #print(y_2)
    #print(y_h)
    print(f'True values y={y} and predicted values y_pred={np.round(y_h)}') 