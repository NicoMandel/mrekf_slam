import numpy as np
from mrekf.binary_bayes import BinaryBayes

"""
    CHECK:
    1. 3B1B
    2. Octomap
    3. Own Quadtree code
        def updateVal(cls, val, pr):

            model = cls.sensor_model
            obs = np.squeeze(model[:, val])
            init_pr = cls.init_prior
            nval = pr + obs - init_pr
            nval = cls.clamping(nval)
            return nval
        
        def clamping(cls, val):
            return np.clip(val, cls.clamp[0], cls.clamp[1])

        
        def getprobabilities(self):
                Function to turn the log-odds back into probabilities (belief)
        
            return (1. - (1. / (1. + np.exp(self.val)))) # if self.val is not None else None

        def getMaxProbability(self):
                Function to get the probability at its maximum index
                only works if val is not none

            return self.getprobabilities()[self.getMaxVal()]


    def getlogprobs(self):
        Function to return the log-odds
        return self.val

    def getMaxVal(self):
        Function to get the argmax - basically whether it is an interesting class or not. 
        return np.argmax(self.val)

    @classmethod
    def instantiate(cls, prior = [0.5, 0.5], sensor_model=[[0.7, 0.3], [0.3, 0.7]], clamp=6):
        
            Method to instantiate class variables to be used for updating the tree. 
            Has to be called before inserting the first element
        prior = np.asarray(prior)
        sensor_model = np.asarray(sensor_model)
        
        # log forms
        cls.sensor_model = np.log( sensor_model / (1. - sensor_model))
        cls.init_prior = np.log(( prior / (1. - prior)))

        # clamping values - are an index        
        cls.clamp = cls.initclamp(clamp)

    @classmethod
    def initclamp(cls, clamp):
        p = cls.init_prior
        for _ in range(clamp):
            model = cls.sensor_model
            obs = np.squeeze(model[:, 1])
            init_pr = cls.init_prior
            p = p + obs - init_pr
        return p



"""

if __name__=="__main__":
    probs = np.array([0.3, 0.7])
    prior = 0.5
    bf = BinaryBayes(probs, prior)
    
    print("Adding 10 1-observations")
    arr = np.ones(10, dtype=int)
    for i in range(arr.size):
        obs = arr[i]
        print("Observation is: {}".format(obs))
        bf.update(obs)
        print("Belief is: {}".format(bf.get_belief()))

    print("Adding 10 0-observations")
    arr = np.zeros(10, dtype=int)
    for i in range(arr.size):
        obs = arr[i]
        print("Observation is: {}".format(obs))
        bf.update(obs)
        print("Belief is: {}".format(bf.get_belief()))

    print("Random Sampling 10 Observations")
    rand_arr = np.random.randint(2,size=10)
    for i in range(10):
        obs = rand_arr[i]
        print("Observation is: {}".format(obs))
        bf.update(obs)
        print("Belief is: {}".format(bf.get_belief()))
