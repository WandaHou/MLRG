import torch
import math
import numpy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def convert(string):
    l = []
    for s in string:
        l.append(int(s))
    return l

def prod(tup):
    result = 1
    for t in tup:
        result *= t
    return result

# diff -> eps
def diff_to_eps(diff):
    batch = diff.shape[0]
    shape = diff.shape[1:]
    diff.unsqueeze(1)
    return torch.cat((diff, torch.zeros_like(diff)), 1).view(batch, 2, *(shape))

# Sn group class projectors
chi_dat = {2: torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
           3: torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]])}

# C4 group irreducible representations
rep_dim = {'A': 1, 'B': 1, 'E': 2}
rep_C4  = {'A': torch.tensor([[1.]]), 
           'B': torch.tensor([[-1.]]), 
           'E': torch.tensor([[0.,-1.],[1.,0.]])}

# bi-adjacency matrices
biadj_dat = {'square': torch.tensor([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]]),
             'cross': torch.tensor([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]]])}

class DynamicBond(torch.nn.Module):
    ''' Dynamic bond with input eps in the graph of an energy model '''
    def __init__(self, chi, repx, repz=None):
        super().__init__()
        self.chi = chi.to(torch.float).to(device) # (n, n, k)
        self.n, _, self.k = self.chi.shape
        self.repx = repx
        self.repz = repz if repz is not None else repx
        self.dx = sum(rep_dim[rep] for rep in self.repx)
        self.dz = sum(rep_dim[rep] for rep in self.repz)
        # construct C4 rotations
        self.C4x = torch.block_diag(*[rep_C4[rep] for rep in self.repx]).unsqueeze(0).to(device) # (1, dx, dx)
        self.C4z = torch.block_diag(*[rep_C4[rep] for rep in self.repz]).unsqueeze(0).to(device) # (1, dz, dz)

    def forward(self, eps): # eps -> (batch, k, dx, dz)
        assert eps.shape[1:] == (self.k, self.dx, self.dz), f'The input eps tensor of {eps.shape[1:]} does not match the expected shape {(self.k, self.dx, self.dz)}.'
        batch = eps.shape[0]
        C4x_inv = self.C4x.mT.unsqueeze(0).expand(batch, *self.C4x.shape)
        C4z = self.C4z.unsqueeze(0).expand(batch, *self.C4z.shape)
        eps_list = [eps]
        for _ in range(3):
            eps = torch.matmul(torch.matmul(C4x_inv, eps), C4z)
            eps_list.append(eps)
        eps_ten = torch.stack(eps_list, 1) # (batch, 4, k, dx, dz)
        return eps_ten
    
class DynamicERBM(torch.nn.Module):
    ''' 
        Equivariant Restricted Boltzmann Machine with Dynamical Bond
            Index for einsum: 
            X -> Lx
            Z -> Lz
            x -> dx
            z -> dz
            i -> Lx dx
            j -> Lz dz
            k -> k
            c -> 4
            b -> batch
            p, q -> n_x, n_z
            s -> samples
    '''
    
    def __init__(self, biadj, dynamic_bond):
        super().__init__()
        self.biadj = biadj.to(torch.float).to(device) # (Lx, Lz, 4)
        self.Lx, self.Lz, _ = self.biadj.shape
        self.dynamic_bond = dynamic_bond
        self.n, self.k = self.dynamic_bond.n, self.dynamic_bond.k
        self.dx, self.dz = self.dynamic_bond.dx, self.dynamic_bond.dz
        self._xz_space = None
        self.clear_cache()
        
    def clear_cache(self):
        self._kernel = None
        self._eps = None
        self._T = None
        
    @property
    def eps(self): # (batch, n, dx, dz)
        return self._eps

    def set_diff(self, diff):
        self.clear_cache() # clear cache before reset eps
        self._eps = diff_to_eps(diff) # convert diff tensor to eps tensor
    
    @property
    def kernel(self):
        if self._kernel is None:
            assert self.eps is not None, f'Before calculating kernel, eps must be specified.'
            eps_ten = self.dynamic_bond(self.eps) # (batch, 4, k, dx, dz)
            # combine bi-partite graph structure to get coupling
            coupling = torch.einsum('XZc,bckxz->bXZkxz', self.biadj, eps_ten) # (batch, Lx, Lz, k, dx, dz)
            coupling = coupling.permute((0,1,4,2,5,3)).reshape(-1, self.Lx * self.dx, self.Lz * self.dz, self.k) # (batch, Lx dx, Lz dz, k)
            # combine group structure to get kernel
            self._kernel = torch.einsum('bijk,pqk->bijpq', coupling, self.dynamic_bond.chi) # (batch, Lx dx, Lz dz, n, n)
        return self._kernel
    
    @property
    def xz_space(self):
        if self._xz_space is None:
            # construct Hilbert space of x
            xs = []
            for i in range(self.n**(self.Lx*self.dx)):
                idx = torch.tensor(convert(reversed(numpy.binary_repr(i, self.Lx*self.dx)))).unsqueeze(-1)
                x = torch.scatter_add(torch.zeros([self.Lx*self.dx, self.n]), -1, idx, torch.ones([self.Lx*self.dx, 1])) # (Lx dx, n)
                xs.append(x)
            xs_shape = [self.n**self.dx for _ in range(self.Lx)] + [self.Lx*self.dx, self.n]
            xs = torch.cat(xs).view(xs_shape).to(device) # (i, j, k, l, Lx dx, n)
            # construct Hilbert space of z
            zs = []
            for i in range(self.n**(self.Lz*self.dz)):
                idx = torch.tensor(convert(reversed(numpy.binary_repr(i, self.Lz*self.dz)))).unsqueeze(-1)
                z = torch.scatter_add(torch.zeros([self.Lz*self.dz, self.n]), -1, idx, torch.ones([self.Lz*self.dz, 1])) # (Lz dz, n)
                zs.append(z)
            zs_shape = [self.n**self.dz for _ in range(self.Lz)] + [self.Lz*self.dz, self.n]
            zs = torch.cat(zs).view(zs_shape).to(device) # square -> (i, j, k, l, Lx dx, n) cross -> (i, Lx dx, n)
            self._xz_space = (xs, zs)
        return self._xz_space
            
    @property
    def T(self): # (batch, n^dx, n^dx, n^dx, n^dx)
        if self._T is None:
            assert self.eps is not None, f'Before calculating rgtensor, eps must be specified.'
            xs, zs = self.xz_space # get Hilber space of x and z
            batch = self.eps.shape[0]
            xs, zs = xs.view(-1, self.Lx*self.dx, self.n), zs.view(-1, self.Lz*self.dz, self.n) # flatten (samples, Lx dx, n)
            Hx, Hz = xs.shape[0], zs.shape[0] # Hx, Hz -> Hilbert space size of x, z
            # trace out zs
            xs, zs = torch.cat([xs for _ in range(zs.shape[0])], 1).view(-1, self.Lx*self.dx, self.n), torch.cat([zs for _ in range(xs.shape[0])], 0).view(-1, self.Lz*self.dz, self.n)
            xs, zs = xs.unsqueeze(0).expand(batch, -1, -1, -1), zs.unsqueeze(0).expand(batch, -1, -1, -1)
            T = (-self.energy(xs, zs).view(batch, Hx, Hz)).exp() # (batch, Hx, Hz)
            T = T.sum(-1).view([batch] + [self.n**self.dx for _ in range(self.Lx)]) # (batch, i, j, k, l)
            self._T = T    
        return self._T
    
    @property
    def G(self):
        lamXi = torch.einsum('bijij->b', self.T)
        lam2Xi = torch.einsum('bijkj,bklil->b', self.T, self.T)
        return lamXi**2/lam2Xi
    
    def scal(self, L=1):
        assert self.T.shape[0] == 1, f'Batch size of the critical point tensor must be 1.'
        T = self.T.squeeze(0)
        size = (T.shape[0])**L # size of transfer matrix
        M = T
        for _ in range(L-1):
            M = torch.tensordot(M, T, dims=([-2], [0])) # construct tensor link
        M = torch.diagonal(M, 0, 0, -2).sum(-1) # trace out left/right legs
        idx1 = torch.arange(0, 2*L, 2) # [0, 2, 4,...]
        idx2 = idx1 + 1 # [1, 3, 5,...]
        idx1, idx2 = idx1.tolist(), idx2.tolist() # index of tensor link legs
        M = torch.permute(M, (*idx1, *idx2))
        idx1, idx2 = torch.arange(0, L).tolist(), torch.arange(L, 2*L).tolist() # [0, 1, 2, 3] [4, 5, 6, 7] .etc
        tran_M = M
        for _ in range(L-1):
            tran_M = torch.tensordot(tran_M, M, dims=(idx2, idx1)) # construct tensor web
        tran_M = tran_M.reshape(size, size)
        eig, Q = torch.linalg.eigh(tran_M) # diagonalize the transfer matrix
        eig = eig.abs()
        return -(eig.log()/(2*math.pi))
            
    def energy(self, x, z): # x -> (batch, samples, Lx dx, n) kernal -> (batch, Lx dx, Lz dz, n, n)
        ez = torch.einsum('bsip,bijpq->bsjq', x, self.kernel) # (batch, samples, Lz dz, n)
        return torch.sum(ez * z, dim=(2,3)) # (batch, samples)
        
    def x_sample_z(self, x):
        ez = torch.einsum('bsip,bijpq->bsjq', x, self.kernel) # (batch, samples, Lz dz, n)
        idx = torch.distributions.Categorical(logits = -ez).sample()
        return torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float)
    
    def z_sample_x(self, z):
        ex = torch.einsum('bsjq,bijpq->bsip', z, self.kernel) # (batch, samples, Lx dx, n)
        idx = torch.distributions.Categorical(logits = -ex).sample()
        return torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float)

    def update_x(self, x, steps=1):
        for _ in range(steps):
            z = self.x_sample_z(x)
            x = self.z_sample_x(z)
        return x

    def update_z(self, z, steps=1):
        for _ in range(steps):
            x = self.z_sample_x(z)
            z = self.x_sample_z(x)
        return z
        
    def sample_x(self, samples, steps=5):
        batch = self.eps.shape[0]
        idx = torch.randint(self.n, (batch, samples, self.Lx * self.dx)).to(device)
        x = torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float) # (batch, samples, Lx dx, n)
        return self.update_x(x, steps=steps)
    
    def sample_z(self, samples, steps=5):
        batch = self.eps.shape[0]
        idx = torch.randint(self.n, (batch, samples, self.Lz * self.dz)).to(device)
        z = torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float) # (batch, samples, Lz dx, n)
        return self.update_z(z, steps=steps)
        
    def sample(self, samples, steps=5):
        x = self.sample_x(samples, steps=steps)
        z = self.x_sample_z(x)
        return x, z

    def cd_x(self, x0, steps=1):
        if steps == 1:
            z = self.x_sample_z(x0)
            x = self.z_sample_x(z)
            ex = torch.einsum('bsjq,bijpq->bsip', z, self.kernel)
            loss = torch.sum(ex * (x0 - x), dim=(2,3))
        else:
            x = x0
            for t in range(steps):
                z = self.x_sample_z(x)
                if t == 0:
                    z0 = z
                x = self.z_sample_x(z)
            loss = self.energy(x0, z0) - self.energy(x, z)
        return loss.mean(-1), x
    
    def cd_z(self, z0, steps=1):
        if steps == 1:
            x = self.z_sample_x(z0)
            z = self.x_sample_z(x)
            ez = torch.einsum('bsip,bijpq->bsjq', x, self.kernel)
            loss = torch.sum(ez * (z0 - z), dim=(2,3))
        else:
            z = z0
            for t in range(steps):
                x = self.z_sample_x(z)
                if t == 0:
                    x0 = x
                z = self.x_sample_z(x)
            loss = self.energy(x0, z0) - self.energy(x, z)
        return loss.mean(-1), z
    
class RGMonotone(torch.nn.Module):
    '''ffn: eps_shape[1:] -> scalar'''
    def __init__(self, diff_shape, dims):
        super().__init__()
        self.shape = diff_shape
        dim_in = prod(self.shape)
        layers = []
        for d in dims:
            layers.append(torch.nn.Linear(dim_in, d))
            layers.append(torch.nn.LayerNorm(d))
            layers.append(torch.nn.GELU())
            dim_in = d
        layers.append(torch.nn.Linear(dims[-1], 1)) # map to a scalar
        self.ffn = torch.nn.Sequential(*layers)

    def forward(self, diff):
        batch = diff.shape[0]
        out = self.ffn(diff.view(batch, -1))
        return out.view(batch, 1)
    
class RGFlow(torch.nn.Module):
    '''perform one-step RG flow by RG monotone'''
    def __init__(self, diff_shape, dims):
        super().__init__()
        self.diff_shape = diff_shape
        self.mono = RGMonotone(diff_shape, dims)
    
    @torch.no_grad()
    def m(self, diff): #output rg monotone
        return self.mono(diff)
    
    def dm(self, diff): # output dm/ddiff
        with torch.set_grad_enabled(True):
            diff.requires_grad_(True)
            m = self.mono(diff)
            dm, *_ = torch.autograd.grad(m, diff, grad_outputs=torch.ones_like(m), create_graph=True)
        return dm
    
    def vec_dm(self, diff): # output vecterized dm/ddiff
        return self.dm(diff.unsqueeze(0)).view(-1)
    
    def forward(self, diff):
        dm = self.dm(diff)
        return diff + dm #+ out_var.exp() * torch.randn_like(out_var).to(diff)
    
    def backward(self, diff):
        dm = self.dm(diff)
        return diff - dm
        
    @torch.no_grad()
    def Newton_step(self, diff, step_size=0.01): # torch.autograd.functional.jacobian cannot do batch-wise, so eps -> (*self.eps_shape)
        assert diff.shape[0] == 1, f'batch size of eps must be 1'
        diff = diff.view(-1)
        J = torch.autograd.functional.jacobian(self.vec_dm, diff)
        diff = diff - step_size * torch.linalg.inv(J) @ self.vec_dm(diff)
        return diff.view(1, *self.diff_shape)
    
class LotusMono(torch.nn.Module):
    ''' 
        Objects:
            DynamicBond: forward batches of eps to eps_ten
            DynamicERBM_square: Equivariant Restricted Boltzmann Machine with square lattice structure
            DynamicERBM_cross: Equivariant Restricted Boltzmann Machine with cross lattice structure
            RGflow: perform rg flow eps0 -> eps1 under time terval (0, 1)
    '''
    def __init__(self, chi, repx, repz, dims=[4, 6], encoder_dims=[6, 12, 6], decoder_dims=[6, 12, 6], VAE_latent_shape=[6]):
        super().__init__()
        self.bond_square = DynamicBond(chi=chi, repx=repx, repz=repz)
        self.bond_cross = DynamicBond(chi=chi, repx=self.bond_square.repz, repz=self.bond_square.repx) # x -> z, z -> x
        self.eps_shape = (self.bond_square.k, self.bond_square.dx, self.bond_square.dz)
        self.mdl_square = DynamicERBM(biadj_dat['square'], self.bond_square)
        self.mdl_cross = DynamicERBM(biadj_dat['cross'], self.bond_cross)
        self.rgflow = RGFlow(diff_shape=self.eps_shape[1:], dims=dims)
        self.VAE = MonoVAE(diff_shape=self.eps_shape[1:], z_shape=VAE_latent_shape, encoder_dims=encoder_dims, decoder_dims=decoder_dims, m=self.rgflow.m)
        
    def clear_cache(self):
        self.mdl_square.clear_cache()
        self.mdl_cross.clear_cache()
        
    def forward_loss(self, samples, steps, diff0=None, batch=10):
        pro_loss = 0.
        if diff0 is None: # propose by MC
            diff0 = self.MC.propose(batch)
            eps0 = diff_to_eps(diff0)
        self.mdl_square.set_diff(diff0)
        dat = self.mdl_square.sample_z(samples, steps)
        diff1 = self.rgflow(diff0)
        diff1 = diff1.permute((0, 2, 1))
        self.mdl_cross.set_diff(diff1)
        loss, _ = self.mdl_cross.cd_x(dat)
        self.clear_cache()
        return loss.mean(-1)
        
    @torch.no_grad()
    def forward(self, diff):
        return self.rgflow(diff)
    
class MonoVAE(torch.nn.Module):
    '''
        Using VAE to approximate the energy function probablity exp(-m)
    
    '''
    def __init__(self, diff_shape, z_shape, encoder_dims, decoder_dims, m):
        super().__init__()
        self.z_shape = z_shape
        self.x_shape = diff_shape
        self.z_dist = torch.distributions.normal.Normal(loc=torch.tensor(0. ,device=device), scale=torch.tensor(1. ,device=device)) # latent variables are from simple Gaussian N(0,1)
        self.encoder_ffn_loc = self.construct_ffn(encoder_dims, self.x_shape, self.z_shape)
        self.encoder_ffn_logsig = self.construct_ffn(encoder_dims, self.x_shape, self.z_shape)
        self.decoder_ffn = self.construct_ffn(decoder_dims, self.z_shape, self.x_shape)
        self.m = m
        
    def construct_ffn(self, dims, shape_in, shape_out):
        dim_in, dim_out = prod(shape_in), prod(shape_out)
        layers = []
        for d in dims:
            layers.append(torch.nn.Linear(dim_in, d))
            layers.append(torch.nn.LayerNorm(d))
            layers.append(torch.nn.GELU())
            dim_in = d
        layers.append(torch.nn.Linear(dims[-1], dim_out))
        return torch.nn.Sequential(*layers)
        
    def sample(self, batch): # sample batches of x
        z = self.z_dist.sample([batch, *self.z_shape])
        x = self.decoder(z)
        return x # shape -> (batch, x_shape)
    
    def decoder(self, z): # given z, decode to x
        x = self.decoder_ffn(z)
        x = x.view([-1, *self.x_shape]) # shape -> (batch, x_shape)
        return x
    
    def encoder(self, x): # given x, encode to z
        x = x.view(x.shape[0], -1)
        mu = self.encoder_ffn_loc(x)
        log_sig = self.encoder_ffn_logsig(x)
        sig = log_sig.exp()
        z = mu + sig * self.z_dist.sample(sig.shape)
        KL = (sig**2/2 + mu**2/2 - log_sig - 1/2).sum(-1)
        return z, KL # KL -> (batch)
    
    def ELBO(self, x, lk=1.): # given x, compute ELBO to approximate log_p(x)
        z, KL = self.encoder(x)
        x_hat = self.decoder(z)
        MSE = ((x - x_hat)**2).sum((1, 2))
        return MSE + lk * KL # shape -> (batch)
    
    def loss(self, x, k=1., lk=1.): # treat ELBO as the approximate log_p(x), compute cross-entropy loss
        p_tgt = torch.exp(k * self.m(x)).view(-1)
        logp_mdl = -self.ELBO(x, lk)
        return (-p_tgt * logp_mdl).mean(-1)
        