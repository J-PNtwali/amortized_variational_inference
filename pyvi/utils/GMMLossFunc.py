import torch
import torch.nn as nn
import numpy as np

def to_triangular(non_zero_tensor: torch.Tensor, dim: int, upper=False) -> torch.Tensor:
    '''
    This function takes as input diagonal and lower-diagonal (resp. upper-diagonal if upper=True) elements of a triangualr matrix (as a 1D torch.Tensor)
    and the dimension, and returns the corresponding triangular matrix (as 2D torch.Tensor)
    
    Input
    =====

        -- non_zero_tensor (torch.tensor): Tensor containing the diagonal and lower-diagonal (resp. upper-diagonal if upper=True) elements of each triangular matrix
                              The code is vectorized, hence `non_zero_tensor` has to be of dimension: (batch_size,  dim * (dim + 1) / 2)
           
        -- dim (int): dimension of each matrix. For batch processing, all the matrices have to be of the same dimension.

        -- upper (bool): equals 'True' for conversion to upper triangular matrices and 'False' lower triangular matrices. Default is `upper = False`.
    
    Output
    ======

        -- t_matrices (torch.tensor): tensor of shape (batch_size, dim, dim) where t_matrices[k,...] is the dim * dim corresponding to non_zero_tensor[k,...]
    '''

    # ensure all dimensions are consistent
    assert dim * (dim + 1) // 2 == non_zero_tensor.shape[-1], \
    'Inconsistent dimensions: the shape of `non_zero_tensor` has to be equal to (batch_size,  dim * (dim + 1) / 2)'


    batch_size = non_zero_tensor.size(0)
    tril_indices = torch.tril_indices(row=dim, col=dim, offset=0)

    if upper:
        # Create the upper triangular matrix
        t_matrices = torch.zeros((batch_size, dim, dim), device=non_zero_tensor.device)
        t_matrices[:, tril_indices[1], tril_indices[0]] = non_zero_tensor
    else:
        # Create the lower triangular matrix
        t_matrices = torch.zeros((batch_size, dim, dim), device=non_zero_tensor.device)
        t_matrices[..., tril_indices[0], tril_indices[1]] = non_zero_tensor
    
    return t_matrices



def exp_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    '''
    This function takes a square matrix as input and returns the same matrix where the diagonal 
    elements have been exponentiated
    '''
    d = matrix.shape[0]
    
    for i in range(d):
        matrix[i, i] = torch.exp(matrix[i, i])
    
    return matrix



#-------------------------------------------------------------------------------------------------------------------
#                       Multi-dimensional, Single Component GMM Loss Function                                      #
#-------------------------------------------------------------------------------------------------------------------
def GaussianLoss(
            target : torch.Tensor,
            mean: torch.Tensor,
            precision_cholesky: torch.Tensor,
            aggr:str=None) -> torch.Tensor:
        '''
            Return the value of the negative log-likelihood loss (NLL) function for a Gaussian model/network, i.e. 
            Gaussian mixture network with a single component.

            Input
            =====
                -- target (Tensor): contains the ``evaluation`` examples where the loss has to be evaluated 

                -- mean (Tensor):  contains the mean of the Gaussian model for each evaluation example
                
                -- precision_cholesky (Tensor): contains the off-diagonal elements and log-scaled diagonal elements of the 
                                                Cholesky factor of the precision matrix of the Gaussian 
                                                model for each evaluation example

            
            Output
            ======
            Gaussian mixture NLL aggregated across evaluation examples

            .
            
        '''
        dim = target.shape[-1]          # dimensionality of the distribution
        
        # write Cholesky factor of each precision matrix as a triangular matrix (note that the diagonal elements are log-scaled)
        precision_cholesky = to_triangular(precision_cholesky.squeeze(), dim, upper=False)
           
        # calculate the log-scaled trace of each Cholesky factor
        trace = torch.vmap(torch.trace)(precision_cholesky)
        
        # put the diagonal elements of each Cholesky factor on the normal scale
        precision_cholesky = torch.vmap(exp_diagonal)(precision_cholesky)

         
        # calculate Q_phi =   L_phi.T @ (theta - mu_phi) 
        Q_phi = torch.matmul(torch.transpose(precision_cholesky, 2, 1), (mean.squeeze() - target.squeeze()).unsqueeze(-1))


        # calculate U_phi = Q_phi.T @ Q_phi
        U_phi =  torch.matmul(torch.transpose(Q_phi, 2, 1), Q_phi)

        # Calculate the negative log-likelihood loss
        loss = 0.5 * (U_phi.squeeze() + dim * np.log(2 * np.pi)) - trace
        
        # aggregate over evaluation examples
        if aggr == 'sum':
            loss = loss.sum()
        elif aggr == 'mean':
            loss = loss.mean()

        return loss
    
 

    
#-------------------------------------------------------------------------------------------------------------------
#                       Multi-dimensional, Multi Component GMM Loss Function                                      #
#-------------------------------------------------------------------------------------------------------------------             
class GaussianMixtureLoss(nn.Module):
    '''
            Evaluate the negative log-likelihood loss (NLL) function for a Gaussian mixture model/network.

            Input
            =====
                -- target (Tensor): contains the ``evaluation`` examples where the loss has to be evaluated 

                -- mean (Tensor):  contains the mean of each component of the Gaussian mixture model for each evaluation example
                
                -- precision_cholesky (Tensor): contains the off-diagonal elements and log-scaled diagonal elements of the 
                                                Cholesky factor of the precision matrix of each component of the Gaussian mixture
                                                model for each evaluation example
                -- coeff (Tensor): contains the mixture coefficients for each evaluation example by row

            
            Output
            ======
            Gaussian mixture NLL aggregated across evaluation examples

            .
            
        '''

    def __init__(self, aggr=None):
        '''
        `aggr` specifies what aggregation measure to apply when evaluating the loss on more than one example at once. Possible values are 
            -- aggr = None ---> no aggregation
            -- aggr = "mean" ---> return the mean loss
            -- aggr = "sum"  ---> return the sum of losses
        '''
        super().__init__()
        self.aggr = aggr
    
    def forward(self,
       target : torch.Tensor,
       mean: torch.Tensor,
       precision_cholesky: torch.Tensor,
       coeff: torch.Tensor=torch.ones(1)
       ) -> torch.Tensor:    
        
       N = target.shape[0]            # number of training examples
       dim = target.shape[-1]         # dimension of the posterior distribution
       K = coeff.shape[-1]            # number of components

       # if the number of components K = 1
       if K==1:
              loss = GaussianLoss(target, mean, precision_cholesky, self.aggr)

       # if the number of components K > 1
       else:
              #-------------------------------------------------------------------------------------------------------------------------
              # 1) Calculate G_1 ~ (N, K) tensor where G_1^(i, k) is the loss from kth component evaluated at the ith training example  #
              #-------------------------------------------------------------------------------------------------------------------------

              # 1.1. replicate target tensor by number of components
              target = target.repeat_interleave(repeats=K, dim=0)     

              # 1.2. reshape mean & precision_cholesky tensors so we can evaluate losses for all components at once                           
              mean = mean.reshape((K * N, dim))        
              precision_cholesky = precision_cholesky.reshape((K * N, precision_cholesky.shape[-1])) 

              # 1.3. evaluate losses by component and training example
              losses = GaussianLoss(target, mean, precision_cholesky, aggr=None).reshape((N, K))
              
              #---------------------------------------------------------------------------------------------------------------------
              # 2)   calculate G_2 =  G_1 @ log(A) where A is the matrix of mixing coefficients and @ is the row-wise dot product  #
              #---------------------------------------------------------------------------------------------------------------------

              losses = torch.bmm(losses.view(N, 1, K), coeff.view(N, K, 1).to(dtype=losses.dtype)).squeeze().to(dtype=torch.float64)

              #-----------------------------------------------------------------------------------------
              # 2)          calculate G_3 = exp(-G_2) where the operation is element-wise              #
              #-----------------------------------------------------------------------------------------
              losses = torch.exp(-losses)
              
              

              #--------------------------------------------------------------------------------
              # 4)            calculate G_4 = log(G_3) and loss J_phi = G_4.sum()             #
              #--------------------------------------------------------------------------------
              losses = - torch.log(losses)

              # aggregate over training examples
              if self.aggr == 'sum':
                     loss = losses.sum()
              elif self.aggr == 'mean':
                     loss = losses.mean()
              else:
                    loss = losses

       return loss
