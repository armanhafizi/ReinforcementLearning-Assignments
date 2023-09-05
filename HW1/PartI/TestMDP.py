from MDP import *

''' Construct simple MDP as described in Lecture 1b Slides 17-18'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)


# delete later
# print(mdp.nActions, mdp.nStates)
# print(mdp.T[1, 1, 3])
# a = np.array([6, 13])
# b = np.array([1, 1])
# print(np.linalg.norm(a-b))
# V_prev = np.zeros(4)
# term = np.amax(R + discount * np.matmul(T, V_prev), axis=0)
# print(term)
# delete later


'''Test each procedure'''
# [V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
# print(V, nIterations, epsilon)
# policy = mdp.extractPolicy(V)
# print(policy)
# V = mdp.evaluatePolicy(np.array([1,0,1,0]))
# print(V)
# [policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
# print(policy,V,iterId)
# [V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
# print(V,iterId,epsilon)
# [policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
# print(policy,V,iterId,tolerance)
