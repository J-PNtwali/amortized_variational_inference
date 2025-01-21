import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from matplotlib import pyplot as plt
from pyvi.utils import GMMLossFunc as lf


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


###################################################################################
#        neural network optimization loop
###################################################################################

def nn_optimizer(model, data_loader, loss_fn, learning_rate = 0.001,
                max_epochs=100, eps=1e-2, echo_after=20, verbose=True, path=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 fused=True, weight_decay=1e-3)
    loss_hist = [0] * max_epochs

    stop = 0   
    
    # optimization loop
    for epoch in range(max_epochs):
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # forward prop
            mean, chol, coeff = model(x_batch)
            # loss
            loss = loss_fn(y_batch, mean, chol, coeff)
            loss.clamp_(max=1e+4)   # clamp loss values
            #print(loss)

            # backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), 2.)
            # update model params
            optimizer.step()
            
            loss_hist[epoch] += loss.item() * y_batch.size(0)

        loss_hist[epoch] /= len(data_loader.dataset)

        if (epoch + 1)  % echo_after == 0 and verbose:
            print('Epoch', epoch + 1,  'Loss', loss_hist[epoch], '\n')
        
        # stopping criteria
        if epoch > 0:
            stop_epoch =  (loss_hist[epoch] > (1 - eps) * loss_hist[epoch - 1])

            if stop_epoch:
                stop += 1
            else:
                stop = 0

        if stop == 3:
            break
        
     # saving the model
    if path:
        torch.save(model, path)
    
    plt.figure(figsize=(8,4))
    plt.plot(torch.arange(epoch+1), loss_hist[:(epoch+1)])
    plt.xlabel('epoch')
    plt.ylabel('loss')
        
    return model


###################################################################################
#       Multivariate Gaussian Mixture neural network class
###################################################################################
class MultivariateGaussianMDN(nn.Module):
    '''
        Multivariate Gaussian Mixture Density Network class for amortized variational inference

        Gaussian mixtures are dense in the space of probability distributions. This motivates their use for posterior density approximation.

        Each mixture is parameterized by the means, Cholesky factors of associated precision matrices and mixture weights.

        The neural network does not output the Cholesky factors directly but rather tensors containing their respective lower diagonal elements.

    '''
    def __init__(self, input_size:int, dim:int=2, K:int=1, hd:int=64, num_hidden=1, sort=False):
        '''
            Input: 
                * input_size: dimension of the input to the neural net, i.e. the number of elements in the observation vector yobs
                * dim:        dimension of the posterior distribution. This is in general the number of parameters in the model.
                * K:          number of mixture components
                * hd:         dimension of each hidden layer
                * num_hidden: number of hidden layers
            Output:
                * mean: tensor of dimensions batchsize X k X dim containing the predicted means
                * chol: tensor of appropriate dimensions containing the the predicted Cholesky factors
                * coeff: tensor of appropriate dimensions containing the predicted mixture component weights
        '''
        super().__init__()

        self.dim = dim
        self.K = K
        self.hd = hd
        self.num_hidden = num_hidden
        self.sort = sort

        # input layer
        self.input_layer = nn.Sequential(nn.Linear(input_size, hd), nn.ELU())
        # hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hd, hd), nn.ELU()) for _ in range(num_hidden) 
            ])
        
        # means of each component
        self.mean = nn.Linear(self.hd, self.K * self.dim)
        
        # Cholesky factors of the precision matrices (diagonal elements are log-scaled)
        self.chol = nn.Linear(self.hd, self.K * self.dim * (self.dim + 1) // 2)

        # mixture weights, non-negative and summing to 1
        self.coeff = nn.Sequential(
            nn.Linear(self.hd, self.K),
            nn.Softmax(dim=1)   
            )          
    
    def forward(self, x):
        if self.sort:
            x = x.sort().values
        # input layer
        x = self.input_layer(x)

        # hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # mean
        mean = self.mean(x)
        mean = mean.reshape((mean.shape[0], self.K, self.dim))

        # Cholecky factor
        chol = self.chol(x)
        chol = chol.reshape((chol.shape[0], self.K, self.dim * (self.dim + 1) // 2))

        # mixture weights
        if self.K > 1:
            coeff = self.coeff(x)
        else:
            coeff = torch.ones(1)
        
        return mean, chol, coeff
    


################################################################################################################
#        simulation-based calibration checks --- plotting
################################################################################################################
def sbc_gaussian(gmmnet, proposal, generator, sample_size=20, n_sim = 1e+4, ecdf=True, ecdf_diff=False, 
                 logscale=None, figsize=(12,4), dpi=100):
    '''
    Perform simulation-based calibration check for a Gaussian mixture network for posterior approximation

    Input:
        -- gmmnet: Gaussian mixture network, with input size given by `sample_size`
        -- proposal: proposal distribution `theta ~ p(theta)`, usually the same as the prior/proposal distribution used for training
                     Note: must have a `sample()` method
        
        -- generator: function that takes parameter values `theta` as input and generate the corresponding simulator model
                       `x ~ p(x|theta)`  as an instance of a class with a `sample` method
        
        -- sample_size: number of iid samples from `x ~ p(x|theta)` for each values of theta

        -- n_sim: number of simulation from the joint distribution: theta ~ p(theta); x ~ p(x|theta)

        -- ecdf: whether to output an eCDF or a histogram plot, default: ecdf=True

        -- ecdf_diff: whether on the y-axis are the `ecdf(w)` values (if True) or `ecdf(w) - w` values (if False).
                        This is ignored if ecdf=False.
        
        -- logscale: (iterable) contains dimensions of the model parameter vector `theta` that are on log-scale
                        note: we use the standard Python counting, starting at 0

    Note: 95% confidence intervals are based on the  Dvoretzky–Kiefer–Wolfowitz inequality (see https://en.wikipedia.org/wiki/Empirical_distribution_function, accessed: 20-05-2024)
    
    Output: SBC plot as a Pyplot figure
    '''

    # draw samples from the prior/proposal  theta ~ p(theta)
    Theta = proposal.sample((n_sim,))

    # draw samples from the model x ~ p(x|theta)
    simulator = generator(Theta)
    X = simulator.sample((sample_size,))
    
    # ensure all dimensions are on the right scale
    if logscale:
        for i in logscale:
            Theta[:,i] = Theta[:,i].log()     # put sigma2 on logscale

    
    # run the gmmnet
    with torch.no_grad():
        mean, chol, coeff = gmmnet(X.to(device))

    n_component, dim = mean.shape[1], mean.shape[2]

    # calculate Cholesky factors
    chol = torch.vmap(lf.exp_diagonal)(lf.to_triangular(chol.view(n_sim * n_component,  dim * (dim + 1) // 2), dim)).view(n_sim, n_component, dim, dim)

    # caclulate precision matrices
    precision = chol @ chol.transpose(2, 3)

    # calculate covariance matrices
    covariance = torch.linalg.inv(precision) 

    # define GMM variational marginal distributions and calculate cdf values for the true parameter values
    W = torch.zeros((n_sim, dim))   # tensor of cdf values
    
    for j in range(dim):
        # mixture weights
        mix = D.Categorical(coeff)
        # mixture components
        comp = D.Normal(mean[:,:,j], torch.sqrt(covariance[:,:,j,j]))
        # define the mixture
        gmm = D.MixtureSameFamily(mix, comp) if n_component > 1 else comp
        # evaluate cdf
        W[:,j] = gmm.cdf(Theta[:,j].to(device))


    if ecdf:
        #=====================================================
        # ECDF plot
        #=====================================================
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Calculate the empirical cumulative distribution function (ECDF)
        eCDF = torch.arange(1, n_sim + 1) / n_sim

        # calculate 95% confidence intervals for the eCDF
        eps = np.sqrt(np.log(2 / 0.05) / (2 * n_sim))
        eCDF_lower, eCDF_upper = eCDF - eps, eCDF + eps

        # exact cdf
        x = np.linspace(0, 1, 100)

        # eCDF for mu
        #===============
        fig.add_subplot(121)

        w = W[:,0].sort().values
        if not ecdf_diff:
            # plot eCDF and true CDF values
            plt.step(w, eCDF, lw=1)
            plt.plot(x, x, 'k--', lw=1)

            # plot 95% confidence bands
            plt.fill_between(w, eCDF_lower, eCDF_upper, color='red', alpha=0.2)

            plt.ylabel(r'$F_{\omega}$')
        else:
            plt.step(w, eCDF - w, lw=1)
            #plt.fill_between(w, eCDF_lower - w, eCDF_upper - w, color='red', alpha=0.1)
            plt.ylabel(r'$F_{\omega} - \omega$')

        plt.xlabel(r'$\omega$')
        plt.title(r'$\mu$')

        # eCDF plot for sigma2
        #======================
        fig.add_subplot(122)

        w = W[:,1].sort().values
        if not ecdf_diff:
            plt.step(w, eCDF, lw=1)
            plt.plot(x, x, 'k--', lw=1)
            # plot 95% confidence bands
            plt.fill_between(w, eCDF_lower, eCDF_upper, color='red', alpha=0.2)

            plt.ylabel(r'$F_{\omega}$')
        else:
            plt.step(w, eCDF - w, lw=1)
            # plot 95% confidence bands
            #plt.fill_between(w, eCDF_lower - w, eCDF_upper - w, color='red', alpha=0.1)
            plt.ylabel(r'$F_{\omega} - \omega$')

        plt.xlabel(r'$\omega$')
        plt.title(r'$\log(\sigma2)$')

        plt.tight_layout()
    else:
        #========================================
        # plot histograms
        #========================================
        fig = plt.figure(figsize=(8, 3), dpi=200)

        # mu
        fig.add_subplot(1,2,1)
        plt.hist(W[...,0], bins=20, density=True, alpha=.6, label=r'$F^{-1}(\mu)$')
        plt.title(r'$\mu$')
        plt.legend(fontsize=7,  markerscale=.5)

        # log(sigma2)
        fig.add_subplot(1,2,2)
        plt.hist(W[...,1], bins=20, density=True, alpha=.6, label=r'$F^{-1}(\log(\sigma^2))$')
        plt.title(r'$\sigma^2$')
        plt.legend(fontsize=7,  markerscale=.5)

        plt.title(r'$\log(\sigma^2)$')
        plt.legend(fontsize=7,  markerscale=.5)

    
    return fig