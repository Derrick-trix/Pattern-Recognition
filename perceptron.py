#part 1
#input
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

lr=0.4
epoch=15000
n=len(x1)

np.random.seed(42)
w1_t=np.random.normal(-1,1)
w2_t=np.random.normal(-1,1)
w3_t=np.random.normal(-1,1)
w0_t=np.random.normal(-1,1)

def meanSqError(w1_t,w2_t,w3_t,w0_t):
    y_pred = expit(w1_t*x1+w2_t*x2+w3_t*x3+w0_t)
    MSE = np.sum((y-y_pred)**2)/(len(y))
    return MSE
for y in logicalfn:
    for i in range(epoch):
        y_pred = expit(w1_t*x1 + w2_t*x2 + w3_t*x3 + w0_t)
        w1_t=w1_t+lr*(1/n*sum(2*(y-y_pred)*(1-y_pred)*(y_pred)*x1))
        w2_t=w2_t+lr*(1/n*sum(2*(y-y_pred)*(1-y_pred)*(y_pred)*x2))
        w3_t=w3_t+lr*(1/n*sum(2*(y-y_pred)*(1-y_pred)*(y_pred)*x3))
        w0_t=w0_t+lr*(1/n*sum(2*(y-y_pred)*(1-y_pred)*(y_pred)*1))
        #if np.mod(i,30) == 0:
            #print(meanSqError(w1_t,w2_t,w3_t,w0_t))
   
    np.set_printoptions(precision=3,suppress=True)
    print("\nGiven Funtion:",functions[func])
    func+=1
    print(f'True values y={y} and predicted values y_pred={y_pred}') 
