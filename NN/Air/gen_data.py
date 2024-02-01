import cantera as ct
import numpy as np
import json
import random
from rich.progress import track

m = 4000 # number of reactions
s_uniform = 500 # number of uniform sample points
T_input = []
P_input = []
Y_input = np.zeros((10000000,5)) # make it large enough
Y_label = np.zeros((10000000,5)) # make it large enough
T_label = []
dt = 1e-7
lamda = 0.1
steps = 10000

y_index = 0
for i in track(range(m), description="gen data..."):
    # sample point distribution
    sample = random.sample(range(1,steps), s_uniform)

    # Data range
    T = np.random.uniform(3000, 6000)
    P = np.random.uniform(0.01*ct.one_atm, 0.1*ct.one_atm)

    Y = np.zeros(5)
    Y[4] = np.random.uniform(0, 1.0)
    Y[1] = np.random.uniform(0, 0.4)
    Y[0] = 10**(-np.random.uniform(0, 20))
    Y[2] = 10**(-np.random.uniform(0, 20))
    Y[3] = 10**(-np.random.uniform(0, 20))
    Y /= sum(Y)

    initial_TPY = T, P, Y

    gas = ct.Solution('./air.yaml')
    gas.TPY = initial_TPY

    r = ct.IdealGasReactor(gas, name='R1')
    sim = ct.ReactorNet([r])

    for tt in range(1, steps):
        if tt in sample:
            T_input.append(np.float64(gas.T))
            P_input.append(np.float64(gas.P))
            Y_input[y_index] = np.float64(gas.Y)

        sim.advance(tt*dt)

        if tt in sample:
            Y_label[y_index] = np.float64(gas.Y)
            y_index += 1
            T_label.append(np.float64(gas.T))

# Additional data
T_input = np.array(T_input)
P_input = np.array(P_input)
T_label = np.array(T_label)

new_len = T_input.shape[0]

input = np.zeros((new_len, 7))
label = np.zeros((new_len, 6))

input[:, 0] = T_input
input[:, 1] = P_input
input[:, 2:7] = np.maximum(Y_input[:new_len, :], 1e-40)

label[:, 0:5] = np.maximum(Y_label[:new_len, :], 1e-40)
label[:, 5] = T_label

n = input.shape[0] 
print("data size = ", n)

# Transformation

bct_input = np.zeros((n,7))
bct_label = np.zeros((n,6))

bct_input[:, 0:2] = input[:, 0:2] # np.vstack((input[:, 0:2], orig_input[:, 0:2]))
bct_input[:,2:7] = (input[:,2:7]**lamda - 1)/lamda # (np.vstack((input[:,2:21], orig_input[:, 2:21]))**lamda - 1) / lamda 

bct_label_old = (label[:, 0:5]**lamda - 1)/lamda # (np.vstack((label[:, :19], orig_label[:, :19]))**lamda - 1) / lamda  
bct_label[:, 0:5] =  (bct_label_old - bct_input[:,2:7]) / dt
bct_label[:, 5] = (label[:, 5] - input[:, 0])/dt

inputs_mean = np.mean(bct_input,axis=0, dtype=np.float64)
inputs_std = np.std(bct_input,axis=0, dtype=np.float64)

labels_mean = np.mean(bct_label,axis=0, dtype=np.float64)
labels_std = np.std(bct_label,axis=0, dtype=np.float64)

norm_inputs = (bct_input - inputs_mean) / inputs_std
norm_labels = (bct_label - labels_mean) / labels_std

data_path = "./norm.json"
normdata = {
    "dt": dt,
    "lambda": lamda,
    "inputs_mean": inputs_mean.tolist(),
    "inputs_std": inputs_std.tolist(),
    "labels_mean": labels_mean.tolist(),
    "labels_std": labels_std.tolist(),
}

with open(data_path, 'w') as json_file:
        json.dump(normdata,json_file,indent=4)

np.savetxt("input.txt", norm_inputs)
np.savetxt("label.txt", norm_labels)