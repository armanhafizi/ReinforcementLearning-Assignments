from re import A
import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q, Q_new = initialQ, np.zeros([self.mdp.nActions,self.mdp.nStates])
        n = np.zeros([self.mdp.nActions,self.mdp.nStates])
        policy = np.zeros(self.mdp.nStates,int)
        rewards = np.zeros(nEpisodes)
        for episode in range(nEpisodes):
            s = s0
            step = 0
            while True:
                if epsilon > 0: # epsilon exploration
                    if np.random.rand() < epsilon:
                        a = np.random.choice(self.mdp.nActions)
                    else:
                        a = policy[s]
                elif epsilon == 0 and temperature > 0: # Boltzman exploration
                    probs = np.exp(Q[:, s]/temperature) / np.sum(np.exp(Q[:, s]/temperature))
                    a = np.random.choice(self.mdp.nActions, 1, p=probs)
                elif epsilon == 0 and temperature == 0: # exploitation
                    a = np.argmax(Q[:, s])
                [r, s_p] = self.sampleRewardAndNextState(s, a)
                rewards[episode] += ((self.mdp.discount ** step) * r)
                n[a, s] += 1
                alpha = 1 / n[a, s]
                Q_new[a, s] = Q[a, s] + alpha * (r + self.mdp.discount * np.amax(Q[:, s_p]) - Q[a, s])
                # policy = self.mdp.extractPolicy(Q[a, :])
                policy = Q.argmax(axis = 0)
                s = s_p
                step += 1
                if step == nSteps:
                    break
                Q = Q_new.copy()
        rewards /= nSteps
        policy = Q.argmax(axis = 0)
        return [Q, policy, rewards]    