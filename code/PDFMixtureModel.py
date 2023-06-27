'''
https://docs.scipy.org/doc/scipy/reference/stats.html#probability-distributions
Now using a continuous probability distribution. Specifically a Von Mises PDF. 

#Source: https://stackoverflow.com/questions/47759577/creating-a-mixture-of-probability-distributions-for-sampling
'''
from scipy.stats import rv_continuous
class PDFMixtureModel(rv_continuous):
    def __init__(self, submodels, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = [w / sum(weights) for w in weights]

    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x) * weight
        return pdf
    
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf