import torch

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


class Bond(torch.nn.Module):
    ''' Directed bond in the graph of an energy model '''
    def __init__(self, chi, repx, repz=None, eps=None):
        super().__init__()
        self.chi = chi.to(torch.float) # (n, n, k)
        self.n, _, self.k = self.chi.shape
        self.repx = repx
        self.repz = repz if repz is not None else repx
        self.dx = sum(rep_dim[rep] for rep in self.repx)
        self.dz = sum(rep_dim[rep] for rep in self.repz)
        if eps is None:
            self.eps = torch.nn.Parameter(torch.randn((self.k, self.dx, self.dz)))
        else:
            assert eps.shape == (self.k, self.dx, self.dz), f'The input eps tensor of {eps.shape} does not match the expected shape {(self.k, self.dx, self.dz)}.'
            self.eps = torch.nn.Parameter(eps) # (k, dx, dz)
        # construct C4 rotations
        self.C4x = torch.block_diag(*[rep_C4[rep] for rep in self.repx]).unsqueeze(0) # (1, dx, dx)
        self.C4z = torch.block_diag(*[rep_C4[rep] for rep in self.repz]).unsqueeze(0) # (1, dz, dz)

    @property
    def eps_ten(self):
        eps = self.eps # (k, dx, dz)
        eps_list = [eps]
        for _ in range(3):
            eps = torch.matmul(torch.matmul(self.C4x.mT, eps), self.C4z)
            eps_list.append(eps)
        eps_ten = torch.stack(eps_list) # (4, k, dx, dz)
        return eps_ten

class ERBM(torch.nn.Module):
    ''' Equivariant Restricted Boltzmann Machine '''
    def __init__(self, biadj, bond):
        super().__init__()
        self.biadj = biadj.to(torch.float) # (Lx, Lz, 4)
        self.Lx, self.Lz, _ = self.biadj.shape
        self.bond = bond
        self.n, self.k = self.bond.n, self.bond.k
        self.dx, self.dz = self.bond.dx, self.bond.dz
        self.clear_cache()
        
    def clear_cache(self):
        self._kernel = None
    
    @property
    def kernel(self):
        if self._kernel is None:
            eps_ten = self.bond.eps_ten # (4, k, dx, dz)
            # combine bi-partite graph structure to get coupling
            coupling = torch.tensordot(self.biadj, eps_ten, dims=([2],[0])) # (Lx, Lz, k, dx, dz)
            coupling = coupling.permute((0,3,1,4,2)).reshape(self.Lx * self.dx, self.Lz * self.dz, self.k) # (Lx dx, Lz dz, k)
            # combine group structure to get kernel
            self._kernel = torch.tensordot(coupling, self.bond.chi, dims=([2],[2])) # (Lx dx, Lz dz, n, n)
        return self._kernel
        
    def energy(self, x, z):
        ez = torch.tensordot(x, self.kernel, dims=([1,2],[0,2]))
        return torch.sum(ez * z, dim=(1,2))
        
    def x_sample_z(self, x):
        ez = torch.tensordot(x, self.kernel, dims=([1,2],[0,2]))
        idx = torch.distributions.Categorical(logits = -ez).sample()
        return torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float)
    
    def z_sample_x(self, z):
        ex = torch.tensordot(z, self.kernel, dims=([1,2],[1,3]))
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
        idx = torch.randint(self.n, (samples, self.Lx * self.dx))
        x = torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float)
        return self.update_x(x, steps=steps)
    
    def sample_z(self, samples, steps=5):
        idx = torch.randint(self.n, (samples, self.Lz * self.dz))
        z = torch.nn.functional.one_hot(idx, num_classes=self.n).to(torch.float)
        return self.update_z(z, steps=steps)
        
    def sample(self, samples, steps=5):
        x = self.sample_x(samples, steps=steps)
        z = self.x_sample_z(x)
        return x, z

    def cd_x(self, x0, steps=1):
        if steps == 1:
            z = self.x_sample_z(x0)
            x = self.z_sample_x(z)
            ex = torch.tensordot(z, self.kernel, dims=([1,2],[1,3]))
            loss = torch.sum(ex * (x0 - x), dim=(1,2))
        else:
            x = x0
            for t in range(steps):
                z = self.x_sample_z(x)
                if t == 0:
                    z0 = z
                x = self.z_sample_x(z)
            loss = self.energy(x0, z0) - self.energy(x, z)
        return loss.mean(), x
    
    def cd_z(self, z0, steps=1):
        if steps == 1:
            x = self.z_sample_x(z0)
            z = self.x_sample_z(x)
            ez = torch.tensordot(x, self.kernel, dims=([1,2],[0,2]))
            loss = torch.sum(ez * (z0 - z), dim=(1,2))
        else:
            z = z0
            for t in range(steps):
                x = self.z_sample_x(z)
                if t == 0:
                    x0 = x
                z = self.x_sample_z(x)
            loss = self.energy(x0, z0) - self.energy(x, z)
        return loss.mean(), z