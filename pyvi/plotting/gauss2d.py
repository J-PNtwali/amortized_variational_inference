import torch
import torch.distributions as torch_dist
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from math import sqrt
from gmm_loss_functions import *

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

####################################################
# define marginal distribution of μ and σ^2
###################################################
def posterior_params(alpha, beta, mu_0, kappa, X):
    N = len(X)
    mu_star = (kappa * mu_0 + N * X.mean()) / (N + kappa)
    alpha_star = alpha + 0.5 * N
    S_bar = (X**2).mean()
    beta_star = 0.5 * (N * S_bar + kappa * mu_0**2 + 2 * beta - (N + kappa) * mu_star**2) 
    
    return mu_star, alpha_star, beta_star


def pdf_marginal_posteriors(alpha, beta, mu_0, kappa, X, 
                               data_mean, data_sigma_2):
    N = len(X)
    # calculate the parameters that define the posterior distribution
    mu_star, alpha_star, beta_star = posterior_params(alpha, beta, mu_0, kappa, X)
    
    # compute the scale and degrees of freedom parameters of the marginal of μ
    nu = 2 * alpha_star                # degrees of freedom
    tau_squared = beta_star / (alpha_star * (kappa + N))   # scale param
    
    
    # compute marginal posterior density of data_mean
    marginal_mu = torch_dist.studentT.StudentT(df=nu, loc=mu_star, scale=sqrt(tau_squared))
    pdf_mean  = torch.exp(marginal_mu.log_prob(data_mean))
    
    # compute marginal posterior density of data_sigma_2
    # use the change of variables formula to compute the pdf of an InverseGamma(alpha_star, beta_star) 
    #  distribution from the pdf of a Gamma(alpha_star, beta_star) distribution
    inverse_marginal_sigma_2 = torch_dist.gamma.Gamma(alpha_star, beta_star)
    pdf_sigma_2 = torch.exp(inverse_marginal_sigma_2.log_prob(1/data_sigma_2) - 2*torch.log(data_sigma_2))
    
    return pdf_mean, pdf_sigma_2    

def pdf_marginal_priors(alpha, beta, mu_0, kappa, 
                               data_mean, data_sigma_2):
    
    # compute the scale and degrees of freedom parameters of the marginal of μ
    tau_squared = beta / (alpha * kappa)   # scale param
    nu = 2 * alpha                    # degrees of freedom
    
    # compute marginal posterior density of data_mean
    marginal_mu = torch_dist.studentT.StudentT(df=nu, loc=mu_0, scale=sqrt(tau_squared))
    pdf_mean  = torch.exp(marginal_mu.log_prob(data_mean))
    
    # compute marginal posterior density of data_sigma_2
    # use the change of variables formula to compute the pdf of an InverseGamma(alpha_star, beta_star) 
    #  distribution from the pdf of a Gamma(alpha_star, beta_star) distribution
    inverse_marginal_sigma_2 = torch_dist.gamma.Gamma(alpha, beta)
    
    pdf_sigma_2 = torch.exp(inverse_marginal_sigma_2.log_prob(1/data_sigma_2) - 2*torch.log(data_sigma_2))
    
    return pdf_mean, pdf_sigma_2    
    
#######################################################
# plot comparisons of marginal distributions 
#######################################################

def compare_marginals_gaussian(model, x, theta, alpha, beta, mu_0, kappa):
    #########################################################
    #### define evaluation range for μ and σ^2  #############
    #########################################################
    # calculate the parameters that define the posterior distribution
    mu_star, alpha_star, beta_star = posterior_params(alpha=alpha, beta=beta, mu_0=mu_0, kappa=kappa, X=x)

    s = np.sqrt((2 * alpha_star + 1) / (2 * alpha_star - 1)) 
    mu_vals = torch.linspace(mu_star - 3 * s, mu_star + 3 * s, 1000)
    sigma_2_vals = torch.linspace(0, 3*alpha_star, 1001)[1:]

    #########################################################
    # evaluate the marginal prior densities
    ########################################################
    mu_pdf_prior, sigma_2_pdf_prior = pdf_marginal_priors(alpha=alpha, beta=beta, mu_0=mu_0, kappa=kappa, 
                                                            data_mean=mu_vals, data_sigma_2=sigma_2_vals)
    
    #########################################################
    # evaluate the exact marginal posterior densities
    ########################################################
    mu_pdf_exact, sigma_2_pdf_exact = pdf_marginal_posteriors(alpha=alpha, beta=beta, mu_0=mu_0, kappa=kappa, X=x,
                                                                data_mean=mu_vals, data_sigma_2=sigma_2_vals)
    
    #########################################################
    # evaluate the approximate marginal posterior densities
    ########################################################
    # calculate mean and covariance params of q_phi
    with torch.no_grad():
        gmm_mean, gmm_cholesky = model(x.to(device))

    gmm_mean, gmm_cholesky = gmm_mean.cpu(), gmm_cholesky.cpu()

    # compute covariance of q_phi from Cholesky factor
    gmm_cholesky = exp_diagonal(triangular(gmm_cholesky.unsqueeze(0), d=gmm_mean.shape, upper=False).squeeze())

    gmm_precision = gmm_cholesky @ gmm_cholesky.T
    gmm_covariance = torch.linalg.inv(gmm_precision)

    mu_gmm = torch_dist.normal.Normal(gmm_mean[0], torch.sqrt(gmm_covariance[0,0]))
    mu_pdf_gmm = torch.exp(mu_gmm.log_prob(mu_vals))

    #  use the change of variable formula to evaluate the approximate pdf of σ^2 from that of log(σ^2)
    logvar_gmm = torch_dist.normal.Normal(gmm_mean[1], torch.sqrt(gmm_covariance[1,1]))
    sigma_2_pdf_gmm = torch.exp(logvar_gmm.log_prob(torch.log(sigma_2_vals))) / sigma_2_vals 
    logsigma_2_pdf_gmm = torch.exp(logvar_gmm.log_prob(torch.log(sigma_2_vals))) 

    #####################################
    #  plotting
    #####################################
    fig = plt.figure(layout='constrained', figsize=(12,8), dpi=200)

    # μ
    fig.add_subplot(221)
    plt.plot(mu_vals, mu_pdf_prior, 'k-.', lw=1.5, label='prior')
    plt.plot(mu_vals, mu_pdf_exact, 'r-', lw=1.5, label='exact posterior')
    plt.plot(mu_vals, mu_pdf_gmm, 'b-', lw=1.5, label='GMM posterior')
    plt.xlabel(r'$\mu$', fontsize=12)
    plt.ylabel(r'$p(\mu$|X)', fontsize=12)
    plt.title('Marginal posterior density of $\mu$', fontsize=14)
    plt.legend(fontsize=12)

    # log(σ^2)
    fig.add_subplot(222)
    plt.plot(torch.log(sigma_2_vals), sigma_2_pdf_prior * sigma_2_vals, 'k-.', lw=2, label='prior')
    plt.plot(torch.log(sigma_2_vals), sigma_2_pdf_exact * sigma_2_vals, 'r-', lw=2, label='exact posterior')
    plt.plot(torch.log(sigma_2_vals), logsigma_2_pdf_gmm, 'b-', lw=2, label='GMM posterior')
    plt.xlabel(r'$\log\sigma^2$', fontsize=12)
    plt.ylabel(r'$p(\log\sigma^2|X)$', fontsize=12)
    plt.title('Marginal posterior density of $\log\sigma^2$', fontsize=14)
    plt.legend(fontsize=12)

    # σ^2
    fig.add_subplot(223)
    plt.plot(sigma_2_vals, sigma_2_pdf_prior, 'k-.', lw=2, label='prior')
    plt.plot(sigma_2_vals, sigma_2_pdf_exact, 'r-', lw=2, label='exact posterior')
    plt.plot(sigma_2_vals, sigma_2_pdf_gmm, 'b-', lw=2, label='GMM posterior')
    plt.xlim(right=alpha_star)
    plt.xlabel(r'$\sigma^2$', fontsize=12)
    plt.ylabel(r'$p(\sigma^2|X)$', fontsize=12)
    plt.title('Marginal posterior density of $\sigma^2$', fontsize=14)
    plt.legend(fontsize=12)

    fig.add_subplot(224)
    labels = ['$\mu$', '$\log(\sigma^2)$']
    z = gauss2d(gmm_mean, gmm_covariance, True, labels)

    plt.show()

#### plot 2d Gaussian density
def gauss2d(mu, sigma, to_plot=False, labels=None):
    w, h = 100, 100

    std = [np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])]
    x = np.linspace(mu[0] - 3 * std[0], mu[0] + 3 * std[0], w)
    y = np.linspace(mu[1] - 3 * std[1], mu[1] + 3 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    if to_plot:
        plt.contourf(x, y, z.T)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.show()

    return z


# MU = np.array([50, 70])
# SIGMA = np.array([[75.0, 10.0], [10.0, 15.0]])
# z = gauss2d(MU, SIGMA, True, ['$\mu$', '$\sigma$'])