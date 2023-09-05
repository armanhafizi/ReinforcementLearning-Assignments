import numpy as np
import MDP

class RL2:
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

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        actionCounter = np.zeros(self.mdp.nActions)
        rewards = []
        for iter in range(nIterations):
            epsilon = 1 / (iter + 1)
            p = np.random.rand()
            if p < epsilon:
                action = np.random.randint(self.mdp.nActions)
            else:
                action = np.argmax(empiricalMeans)
            [reward, _] = self.sampleRewardAndNextState(0, action)
            empiricalMeans[action] = (actionCounter[action]*empiricalMeans[action]+reward) / (actionCounter[action]+1)
            actionCounter[action] += 1
            rewards.append(reward)
        return empiricalMeans, np.array(rewards)

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        rewards = []
        for iter in range(nIterations):
            for action in range(self.mdp.nActions):
                empiricalMeans[action] = np.mean(np.random.beta(prior[action, 0], prior[action, 1], k))
            action = np.argmax(empiricalMeans)
            [reward, _] = self.sampleRewardAndNextState(0, action)
            # line below is in the algorithm but will not affect the future iterations
            # but it might make the reward greater than 1 in some cases. That is why I commented it.
            # empiricalMeans[action] += reward
            prior[action, int(not(reward))] += 1
            rewards.append(reward)
        return empiricalMeans, np.array(rewards)

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        actionCounter = np.zeros(self.mdp.nActions)
        rewards = []
        for iter in range(nIterations):
            if iter == 0:
                action = np.argmax(empiricalMeans)
            else:
                action = np.argmax(empiricalMeans + np.sqrt(2*np.log(iter)/actionCounter))
            [reward, _] = self.sampleRewardAndNextState(0, action)
            empiricalMeans[action] = (actionCounter[action]*empiricalMeans[action]+reward) / (actionCounter[action]+1)
            actionCounter[action] += 1
            rewards.append(reward)
        return empiricalMeans, np.array(rewards)