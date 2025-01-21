# Helper tools for the Gaussian with unknown mean and variance posterior approximation experiment

import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from math import sqrt
from pyvi.utils import GMMLossFunc as lf

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

plt.style.use('fivethirtyeight')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})




#============================================================================================
# define the Gaussian - InverseGamma conjugate prior for the mean and variance parameters 
#============================================================================================

class GaussianInverseGamma():
    '''
        Defines Gaussian - InverseGamma (GaussianIG) prior class

        The Gaussian-InverseGamma distribution is a conjugate prior for the Gaussian observation model:
            y ~ N(mu, sigma2)
            where both `mu` and `sigma2` are unknown 

        The GaussianIG prior is defined by specifying a conditional Gaussian distribution on `mu` given `sigma2` and a marginal InverseGamma distribution for `sigma2` in the following way:

            mu|sigma2 ~ N(mu_0, sigma2 / kappa),     kappa > 0
               sigma2 ~ InverseGamma(alpha, beta),  alpha > 0, beta > 0
        
        Each instance of the GaussianIG class takes 2 parameters (hyperparameters for the model): 
            - params_mu: {'mu_0': mu_0, 'kappa': kappa}  is a dictionary containing the values of `mu_0` and `kappa`
            - params_sigma2: {'alpha': alpha, 'beta': beta} is a dictionary containing the values of `alpha` and `beta`

        The class implements 2 methods: the `sample` and `log_prob` methods for sampling from the prior and log-probability density calculation, respectively.

        # Example
        >>> params_mu = {'mu_0': 0.0, 'kappa': 1.0}
        >>> params_sigma2 = {'alpha': 1.0, 'beta': 1.0}
        >>> prior = GaussianIG(params_mu, params_sigma2)
        >>> n = 10, theta = prior.sample((n,))     # sampling, n examples
                                                   # note that `prior.sample()` is equivalent to `prior.sample((1,))`
        >>> print(theta)
            tensor([[0.0390, 2.6079],
                    [1.1483, 0.6664],
                    [0.0826, 0.5726]])
        >>> logprob = prior.log_prob(theta)        # log densities
        >>> print(logprob)
            tensor([-3.6990, -2.3942, -1.2773])
    '''
    def __init__(self, params_mu, params_sigma2):
        # recode hyperparameters. This allows for the object to be updated with no need of re-instantiation
        self.params = {'mu':params_mu, 'sigma2':params_sigma2}
        
        # define the InverseGamma marginal distribution for `sigma2` 
        self.sigma2 = torch.distributions.inverse_gamma.InverseGamma(self.params['sigma2']['alpha'],
                                                                          self.params['sigma2']['beta'])
        
        # define the conditional Gaussian distribution for `mu` given  `sigma2`
        self.mu = lambda sigma2: torch.distributions.normal.Normal(loc=self.params['mu']['mu_0'], 
                                                                   scale=torch.sqrt(sigma2 / self.params['mu']['kappa']))
        
        # marginal distribution for mu
        self.mu_marginal = torch.distributions.StudentT(df=2*self.params['sigma2']['alpha'],
                                                        loc=self.params['mu']['mu_0'],
                                                        scale=sqrt(
                                                            self.params['sigma2']['beta'] / (self.params['sigma2']['alpha'] * self.params['mu']['kappa'])
                                                        ))
        
    def sample(self, shape: torch.Size = ()):
        '''
        Method for sampling `(mu, sigma2)` from the Gaussian-InverseGamma distribution. This is 2-step approach:
            (1) sample: sigma2 ~ InverseGamma(alpha, beta)
            (2) sample: mu ~ N(mu_0, sigma2 / kappa)

        '''
        sigma2 = self.sigma2.sample(shape)              # (1) sample: sigma2 ~ InverseGamma(alpha, beta)
        mu = self.mu(sigma2).sample()                   # (2) sample: mu ~ N(mu_0, sigma2 / kappa)
 
        theta = torch.stack([mu, sigma2], dim=-1)       # combine
        return theta
        
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        '''
        Method for log-density evaluation for the Gaussian-InverseGamma distribution based on the following formula:
        
            log(p(mu, sigma2)) = log(p(mu|sigma2) * p(sigma2))
                            = log(p(mu|sigma2)) + log(p(sigma2))
        
        Note that the code is vectorized so that `theta` can contain multiple observations 
        '''
        logprob = self.mu(theta[...,1]).log_prob(theta[...,0]) + self.sigma2.log_prob(theta[...,1]) 
        
        return logprob


#=======================================================================================================================
#  define the exact posterior distribution over (mu, sigma2) when the prior is GaussianInverseGamma
#=======================================================================================================================

def GaussianInverseGammaPosterior(prior, yobs):
    ''''
    Defines the exact posterior distribution over parameters (mu, sigma2) of the Gaussian model with unknown mean and variance 
    when  the prior distribution is an instance of the GaussianInverseGamma class

    Input: 
        - prior: an instance of the GaussianInverseGamma class
        - yobs:  observed (or simulated) data 

    Output: posterior distribution, an instance of the GaussianInverseGamma class
    '''

    # prior hyper params
    params_mu, params_sigma2 = prior.params['mu'].copy(), prior.params['sigma2'].copy()
    
    # calculate corresponding posterior hyper params
    mu_0, kappa = params_mu['mu_0'], params_mu['kappa']
    n = yobs.shape[0]
    s2 = ((yobs - yobs.mean())**2).sum() / (n-1)

    params_mu['mu_0'] = (kappa * mu_0 + yobs.sum()) / (kappa + n)
    params_mu['kappa'] = kappa + n

    params_sigma2['alpha'] = params_sigma2['alpha'] + 0.5 * n
    params_sigma2['beta'] = params_sigma2['beta'] + 0.5 * (n-1) * s2 + 0.5 * kappa * n * (yobs.mean() - mu_0)**2 / (kappa + n)

    return GaussianInverseGamma(params_mu, params_sigma2)

    


#============================================================================
#  define the 1d Gaussian simulator with parameters theta = [mu, sigma2]
#============================================================================

class Gaussian1DSimulator():
    '''
        Defines a class for the Gaussian probability model given by:
            y ~ N(mu, sigma2)       # observation model
            theta = [mu, sigma2]    # model parameters
    '''
    def __init__(self, theta: torch.Tensor):
        '''
        - Define the distribution `N(mu, sigma2)`
        - Note that the code is vectorized so that `theta` can be a 2-d tensor where each row contains `theta_i = (mu_i, sigma_i)` defining a single model
        - sampling and log-density evaluation are similarly vectorized
        '''
        self.theta = theta

        self.model = torch.distributions.normal.Normal(self.theta[...,0], torch.sqrt(self.theta[...,1]))
        
    def sample(self, shape: torch.Size = ()):
        y = self.model.sample(shape)

        return y.T
        
    def log_prob(self, y):
        logprob = self.model.log_prob(y) 
        
        return logprob
    


def GenerateSimulator(theta):
    '''
    Wrapper around the Gaussian1DSimulator class. 

    Input: 
        -- theta: tensor containing values of mu and sigma2
    
    Output: An instance of the Gaussian1DSimulator class, with self.theta = theta
    '''
    return Gaussian1DSimulator(theta)




#==========================================================================================================
# compare variational posterior with exact posterior distributions -- histogram plots
#==========================================================================================================
def posterior_compare(gmmnet, prior, yobs, theta, n_samples=1000, n_sim=1,
                       labels=[r'$\mu$', r'$\sigma^2$'], title=None,
                         show_prior=False, figsize=(12,4), dpi=100):
    
    with torch.no_grad():
        mean, chol, coeff = gmmnet(yobs.unsqueeze(0).to(device))
       # mean, chol, coeff = mean.cpu(), chol.cpu(), coeff.cpu()

    # derive precision matrices
    n_component, dim = mean.shape[1], mean.shape[2]

    # calculate Cholesky factors
    chol = torch.vmap(lf.exp_diagonal)(lf.to_triangular(chol.view(n_sim * n_component,  dim * (dim + 1) // 2), dim)).view(n_sim, n_component, dim, dim)

    # calculate precision matrices
    precision = chol @ chol.transpose(2, 3)

    # calculate covariance matrices
    covariance = torch.linalg.inv(precision)

    mix = D.Categorical(coeff)

    ####  EXACT POSTERIOR ###
    posterior = GaussianInverseGammaPosterior(prior=prior, yobs=yobs)
    samples = posterior.sample((n_samples,))

    #### PLOTTING ####
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # define Gaussian mixture distribution

    for k in range(dim):
        ####   APPROXIMATE MARGINAL POSTERIOR ###
        # mixture components
        comp = D.Normal(loc=mean[:,:,k], scale=torch.sqrt(covariance[:,:,k,k]))
        # define the mixture
        gmm = D.MixtureSameFamily(mix, comp) if n_component > 1 else D.Normal(mean[:,:,k].squeeze(), torch.sqrt(covariance[:,:,k,k]).squeeze())

        #### SAMPLING ###
        Theta_prior = prior.mu_marginal.sample((n_samples,)).sort(dim=0).values if k==0 else prior.sigma2.sample((n_samples,)).sort(dim=0).values      # prior distribution
        Theta_exact = posterior.mu_marginal.sample((n_samples,)).sort(dim=0).values if k==0 else posterior.sigma2.sample((n_samples,)).sort(dim=0).values  # exact posterior distribution
        Theta_gmm = gmm.sample((n_samples,)).squeeze().sort(dim=0).values if k==0 else gmm.sample((n_samples,)).exp().squeeze().sort(dim=0).values  # GMM approximate posterior

        
        #### EVALUATE DENSITIES ####
        # prior
        pdf_prior = prior.mu_marginal.log_prob(Theta_prior).exp() if k==0 else prior.sigma2.log_prob(Theta_prior).exp()
        # exact posterior
        pdf_posterior_exact = posterior.mu_marginal.log_prob(Theta_exact).exp() if k==0 else posterior.sigma2.log_prob(Theta_exact).exp()
        # gmm posterior
        pdf_posterior_gmm = gmm.log_prob(Theta_gmm).exp() if k==0 else gmm.log_prob(Theta_gmm.log()).exp() / Theta_gmm
        
        # true value & MLE    
        theta_k_true =  theta[...,k]
        mle =  yobs.mean() if k == 0 else yobs.var()

        fig.add_subplot(1, dim, k+1)
        plt.plot(Theta_gmm.cpu(), pdf_posterior_gmm.cpu(), 'b-', lw=1.5, label='Posterior - NPE')
        plt.plot(Theta_exact, pdf_posterior_exact, 'g-', lw=1.5, label='Posterior - Exact')
        
        if show_prior:
            plt.plot(Theta_prior, pdf_prior, 'k--', lw=1.5, label='Prior')

        plt.vlines(x=theta_k_true, ymin=plt.axis()[2], ymax=plt.axis()[3],  colors='black', lw=1, label='True value')
        plt.vlines(x=mle, ymin=plt.axis()[2], ymax=plt.axis()[3], linestyles='dashed', colors='black', lw=1, label='MLE')
    
            
        plt.xlabel(r'$\theta$', fontsize=12)
        plt.ylabel(r'$f_\theta(\theta)$', fontsize=12)
        plt.legend(fontsize=12)
        plt.title(labels[k])

    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

    
