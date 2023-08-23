import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

class Errors():
    def __init__(self, predictions, points):
        self.predictions = predictions
        self.points = points 

        self.no_errors = points - predictions
        self.errors = np.sort(points - predictions)

        self.N = np.size(self.errors)

        self.mean = np.mean(self.errors) # not used

        self.sigma = np.sqrt(np.mean(abs(self.errors) ** 2))

        self.Lik = np.prod(sp.stats.norm.pdf(self.errors, loc = 0, scale = self.sigma)) # not used
        self.logLik = np.sum(
                np.log( sp.stats.norm.pdf(self.errors, loc = 0, scale = self.sigma) )
            )

    # Graphical Test

    def QQ(self, title, filename, path):
        u = np.sort( np.random.uniform(size = self.N) )
        q = sp.stats.norm.ppf(u,loc = 0, scale = self.sigma)
        plt.plot(q,self.errors, 'o')

        min_ = min(np.min(q), np.min(self.errors))
        max_ = max(np.max(q), np.max(self.errors))

        fig, ax = plt.subplots(
            figsize = (6,4),
            tight_layout = True
        )

        ax.plot(q,self.errors, '.', color = "b")
        ax.plot(
            [min_,max_], [min_,max_], 
            '-', color = 'k')

        ax.set_xlabel("Observations", fontsize = 14)
        ax.set_ylabel("Normal quantiles", fontsize = 14)

        ax.set_title(title)
        ax.grid(True)

        filename = filename + '.png'
        plt.savefig(filename)
        os.replace(filename, 
            os.path.join(path, filename)
        )

        plt.close( fig )

    def pdfPlot(self, title, filename, path):
        prob = 0.01
        N = 100

        q0 = sp.stats.norm.ppf(prob, loc = 0, scale = self.sigma)
        q0 = min(q0, np.min(self.errors))

        q1 = sp.stats.norm.ppf(1-prob, loc = 0, scale = self.sigma)
        q1 = max(q1, np.max(self.errors))

        qs = np.linspace(q0, q1, num = N)
        d = sp.stats.norm.pdf(qs, loc = 0, scale = self.sigma)

        fig, ax = plt.subplots(
            figsize = (6,4), tight_layout = True
        )

        ax.set_title(title, fontsize = 16)

        ax.plot(qs, d, '-', label = "Density function")
        ax.plot(self.errors, np.zeros(len(self.errors)), 'o')

        ax.set_xlabel('observations', fontsize = 14)
        ax.set_xlabel('density function', fontsize = 14)

        filename = filename + ".png"
        ax.legend( ["Density function"], loc = "best")
        ax.grid(True)

        plt.savefig(filename)
        os.replace(filename, os.path.join(path, filename))

        plt.close( fig )

    def varianceEvolution(self, title = "", filename = "", path = ""):
        n = np.linspace(1, self.N, num = self.N)

        fig, ax = plt.subplots(
            figsize = (6, 4),
            tight_layout = True
        )

        ax.set_title(title, fontsize = 16)

        ax.plot(n, self.no_errors, 'o')
        ax.plot([n[0], n[-1]], [0,0], '-')

        ax.set_xlabel("Observation number", fontsize = 14)
        ax.set_ylabel("Value", fontsize = 14)
        ax.grid(True)

        filename = filename + ".png"
        plt.savefig(filename)

        os.replace(filename, os.path.join(path, filename))
        plt.close(fig)

    def NvsN_1(self, title = "", filename = "", path = ""):
        n = np.linspace(1, self.N, num = self.N)

        fig, ax = plt.subplots(
            figsize = (6, 4),
            tight_layout = True
        )

        en = self.no_errors[1:]
        en_1 = self.no_errors[:-1]

        ax.set_title(title, fontsize = 16)

        ax.plot(en, en_1, 'o')

        ax.set_xlabel("n-th error", fontsize = 14)
        ax.set_ylabel("(n-1)-th error", fontsize = 14)
        ax.grid(True)

        filename = filename + ".png"
        plt.savefig(filename)

        os.replace(filename, os.path.join(path, filename))
        plt.close(fig)

    def Graphs(self, directory, experiment, split, variable, key, model):
        self.varianceEvolution(
            f'Variance Evolution in errors:\n{experiment}-{split}-{variable}-{key}-{model}',
            filename = f'{experiment}-{split}-{variable}-{key}-{model}-variance',
            path = directory
        )
        self.pdfPlot(
            f'Density function:\n{experiment}-{split}-{variable}-{key}-{model}',
            filename = f'{experiment}-{split}-{variable}-{key}-{model}-density',
            path = directory
        )
        self.QQ(
            f'QQ-plot for normal model:\n{experiment}-{split}-{variable}-{key}-{model}',
            filename = f'{experiment}-{split}-{variable}-{key}-{model}-QQ',
            path = directory
        )
        self.NvsN_1(
            f'N-th vs (N-1)-th residuals:\n{experiment}-{split}-{variable}-{key}-{model}',
            filename = f'{experiment}-{split}-{variable}-{key}-{model}-NvsN_1',
            path = directory
        )

    
    # Numerical Values

    def normalityTest(self):
        if self.N < 8:
            print("Error in normality Test: too few samples")
            return float('NaN')
        res = sp.stats.normaltest(self.errors)
        return res.pvalue

    def homocedasticityLevene(self, Nsplits = 2):
        try:
            splitted = np.array_split(self.errors, Nsplits)
            res = sp.stats.levene(
                * splitted
            )
            return res.pvalue
        except:
            return float('NaN')
    
    def homocedasticityBartlett(self, Nsplits = 2):
        try:
            splitted = np.array_split(self.errors, Nsplits)
            res = sp.stats.bartlett(
                * splitted
            )
            return res.pvalue
        except:
            return float('NaN')


    def AIC(self):
        return 2 * 1 - 2 * self.logLik

    def BIC(self):
        return 2 * 1 * np.log(self.N) - 2 * self.logLik

    def R2(self):
        mn = np.mean(self.points)

        ss_res = np.sum(self.errors ** 2)
        ss_tot = np.sum((self.points - mn)**2)

        return 1 - ss_res / ss_tot 


def main():
    N = 100
    pred = np.linspace(0, 3, num = N)
    sigma = 1
    points = pred + np.random.default_rng().normal(0,sigma, N)
    # points = pred + np.random.exponential(sigma, N)

    es = Errors(pred, points)
    # es.QQcent()
    # es.QQ()
    # es.pdf_comparison()
    es.varianceEvolution(
        f'Variance Evolution in errors expected',
        filename = "Expected-variance",
        path = "src/juan/"
    )

    # print(es.BICcent())
    # print(es.BIC())

    print(es.normalityTest())

if __name__ == '__main__':
    main()