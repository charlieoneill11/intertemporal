from tkinter import N
import numpy as np
import pandas as pd

class Dataset:

    def __init__(self, sir_range, ldr_range, delay_range, train_size=50, test_size=25):
        self.sir_range = sir_range 
        self.ldr_range = ldr_range
        self.delay_range = delay_range
        self.train_size = train_size
        self.test_size = test_size

    def generate_data(self, train=True):
        if train: 
            n = self.train_size
            name = "train"
        else: 
            n = self.test_size
            name = "test"
        sir = np.random.randint(self.sir_range[0], self.sir_range[1], n)
        ldr = sir + np.random.randint(self.ldr_range[0], self.ldr_range[1], n)
        delay = np.random.randint(self.delay_range[0], self.delay_range[1], n)
        answers = ["NA" for _ in range(n)]
        data_dict = {'SIR': sir, 'LDR': ldr, 'Delay': delay, 'Answer': answers}
        df = pd.DataFrame(data_dict)
        df.to_csv(f"~/intertemporal/data/templates/{name}_template.csv", index=False)

if __name__ == "__main__":
    generate = Dataset(sir_range = [5, 1000], ldr_range=[10, 400], delay_range = [20, 500])
    generate.generate_data()
    generate.generate_data(train=False)