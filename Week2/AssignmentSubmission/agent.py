import numpy as np
import random

# file_path = "../Assignment/MDPs/mdp-2-2.txt"
file_path = "../Assignment/MDPs/mdp-10-5.txt"
with open(file_path, 'r') as file:
    badlines = file.readlines()

lines = [line.strip() for line in badlines]

splitLines = []
for line in lines:
    splitLines.append(line.split(' '))
    
numLines = len(lines)
numStates = int(splitLines[0][1])
numActions = int(splitLines[1][1])
gamma = float(splitLines[numLines - 1][len(splitLines[numLines - 1]) - 1])

P = {}
for i in range(2, numLines - 1):
    state = int(splitLines[i][1])
    action = int(splitLines[i][2])
    store = (int(splitLines[i][3]), float(splitLines[i][4]), float(splitLines[i][5]))
    if state in P:
        if action in P[state]:
            P[state][action].append(store)
        else:
            P[state][action] = []
            P[state][action].append(store)
    else:
        P[state] = {}
        P[state][action] = [store]

def value_iteration(P, gamma = 1.0, theta = 1e-10):
    V = np.zeros(len(P), dtype = np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype = np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for next_state, reward, prob in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state])
        if np.max(np.abs(V - np.max(Q, axis = 1))) < theta:
            break
        V = np.max(Q, axis = 1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis = 1))} [s]
    for i in range(len(V)):
        V[i] = round(V[i], 6)
    return V, pi
V, pi = value_iteration(P, gamma)
bestActions = []
for s in range(numStates):
    bestActions.append(pi(s))

with open('output.txt', 'w') as file:
    for i in range(numStates):
        print(f'{V[i]} {bestActions[i]}', file = file)