import numpy as np
from mrekf.binary_bayes import BinaryBayes


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
