import numpy as np
from scipy.optimize import minimize

class ParameterFit:
    
    def __init__(self, train, test):
        self.train = train
        self.test = test
    
    def get_prob_choice(self, k, sir, ldr, delay, real_choice):
        if real_choice == 1: 
            p_choice = np.exp(ldr/(1 + k*delay)) / (np.exp(sir) + np.exp(ldr/(1 + k*delay)))
        else:
            p_choice = (1 -  (np.exp(ldr/(1 + k*delay)) / (np.exp(sir) + np.exp(ldr/(1 + k*delay)))))
        return p_choice

    def generate_log_likelihood(self, current_k, train):
        # define vector that will store the probability that the model chooses
        choice_probs = np.zeros((len(train),1))
        for j in range(len(train)):
            # load the choice probability vector for every choice
            choice_probs[j] = self.get_prob_choice(current_k, train.SIR.iloc[j], train.LDR.iloc[j], 
                                train.Delay.iloc[j], train.Answer.iloc[j])
        # take sum of logs and negative to work within minimisation framework
        return (-1)*np.sum(np.log(choice_probs))

    def simulate_choice(self, row, k):
        value = (row["LDR"]/(1+k*row["Delay"])) - row["SIR"]
        return 1 if value >= 0 else 0
    
    def fit(self):
        k_0 = 0.001
        res = minimize(self.generate_log_likelihood, k_0, args=(self.train), method='BFGS')
        k_fit = res.x[0]
        preds = self.test.apply(self.simulate_choice, k=k_fit, axis=1)
        return preds