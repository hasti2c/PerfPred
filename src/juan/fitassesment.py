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

        self.mean = np.mean(self.errors)

        self.sigmacent = np.sqrt(np.mean(abs(self.errors) ** 2))
        self.sigma = np.sqrt(np.var(self.errors))

        self.LikCent = np.prod(sp.stats.norm.pdf(self.errors, loc = 0, scale = self.sigmacent))
        self.logLikCent = np.sum(
                np.log( sp.stats.norm.pdf(self.errors, loc = 0, scale = self.sigmacent) )
            )

        self.Lik = np.prod(sp.stats.norm.pdf(self.errors, loc = self.mean, scale = self.sigma))
        self.logLik = np.sum(
                np.log( sp.stats.norm.pdf(self.errors, loc = self.mean, scale = self.sigma) )
            )

    def QQcent(self, title, filename, path):
        u = np.sort( np.random.uniform(size = self.N) )
        q = sp.stats.norm.ppf(u,loc = 0, scale = self.sigmacent)
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

    def QQ(self, title = "", filename = "", path = ""):
        u = np.sort( np.random.uniform(size = self.N) )
        q = sp.stats.norm.ppf(u,loc = self.mean, scale = self.sigma)
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

    def pdf_comparison(self, title, filename, path):
        prob = 0.01
        N = 100
        q0 = min(
                    sp.stats.norm.ppf(prob, loc = self.mean, scale = self.sigma),
                    sp.stats.norm.ppf(prob, loc = 0, scale = self.sigmacent),
                )
        q0 = min(q0, np.min(self.errors))
        q1 = max(
                    sp.stats.norm.ppf(1-prob, loc = self.mean, scale = self.sigma),
                    sp.stats.norm.ppf(1-prob, loc = 0, scale = self.sigmacent)
                )
        q1 = max(q1, np.max(self.errors))

        qs = np.linspace(q0, q1, num = N)
        d1 = sp.stats.norm.pdf(qs, loc = self.mean, scale = self.sigma)
        d2 = sp.stats.norm.pdf(qs, loc = 0, scale = self.sigmacent)

        fig, ax = plt.subplots(
            figsize = (6,4),
            tight_layout = True
        )

        ax.set_title(title, fontsize = 16)

        ax.plot(qs, d1, '-', label = "Best normal model")
        ax.plot(qs, d2, '-', label = "Centered normal model")
        ax.plot(self.errors, np.zeros(len(self.errors)), 'o')

        ax.set_xlabel('observations', fontsize = 14)
        ax.set_xlabel('density function', fontsize = 14)

        filename = filename + ".png"
        ax.legend( ["Best normal model", "Centered normal model"], loc = "best")
        ax.grid(True)

        plt.savefig(filename)
        os.replace(filename, os.path.join(path, filename))

        plt.close( fig )

    def normalityTest(self):
        if self.N < 8:
            print("Error in normality Test: too few samples")
            return float('NaN')
        res = sp.stats.normaltest(self.errors)
        return res.pvalue

    def homocedasticityLevene(self, Nsplits = 2):
        splitted = np.array_split(self.errors, Nsplits)
        res = sp.stats.levene(
            * splitted
        )
        return res.pvalue
    
    def homocedasticityBartlett(self, Nsplits = 2):
        splitted = np.array_split(self.errors, Nsplits)
        res = sp.stats.bartlett(
            * splitted
        )
        return res.pvalue

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

    def AICcent(self):
        return 2 * 1 - 2 * self.logLikCent

    def AIC(self):
        return 2 * 2 - 2 * self.logLik

    def BICcent(self):
        return 2 * 1 * np.log(self.N) - 2 * self.logLikCent

    def BIC(self):
        return 2 * 2 * np.log(self.N) - 2 * self.logLik

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