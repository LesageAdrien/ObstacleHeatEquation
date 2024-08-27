import numpy as np
import scipy.sparse as sps

def dx(f):
    res = np.zeros_like(f)
    res[1:] += f[:-1]
    res[:-1] -= f[1:]
    return res

def L(N):
    return sps.diags([np.ones(N-1), -2*np.ones(N), np.ones(N-1)], [-1,0,1], dtype = float, format = "csr")*N**2
class Simulator:
    def __init__(self, Nx, Ny, circlepos = (0.5, 0.5), circleradius = 0.1):
        self.Nx, self.Ny = Nx, Ny
        self.x, self.y = np.linspace(0, 1, Nx, endpoint=False), np.linspace(0, 1, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)



        self.border_condition_types = ('neu', 'dir', 'neu', 'dir') #left, top, right, bottom
        self.border_condition_values = (0, 1, 0, 0)

        self.U = np.zeros_like(self.X).astype(float)
        self.LAMBDA = np.zeros_like(self.U)

        self.PHI = (((self.X - circlepos[0])**2 + (self.Y - circlepos[1])**2) < circleradius ** 2 ).astype(float)
        self.D = np.sqrt((dx(self.PHI) * self.Nx) ** 2 + (dx(self.PHI.T).T * self.Ny) ** 2)
        self.s = 1


        self.eye = sps.eye(self.Nx*self.Ny)
        L1 = L(self.Nx)
        L2 = L(self.Ny)

        self.S = np.zeros_like(self.PHI).astype(float)

        if self.border_condition_types[0] == 'dir':
            self.S[0] += self.border_condition_values[0] * self.Nx**2
        else:
            L1[0, 1] += 1*self.Nx**2
            self.S[0] += self.border_condition_values[0]*self.Nx

        if self.border_condition_types[2] == 'dir':
            self.S[-1] += self.border_condition_values[2] * self.Nx**2
        else:
            L1[-1, -2] += 1*self.Nx**2
            self.S[-1] += - self.border_condition_values[2] * self.Nx

        if self.border_condition_types[1] == 'dir':
            self.S[:, 0] += self.border_condition_values[1] * self.Ny**2
        else:
            L2[0, 1] += 1 * self.Ny ** 2
            self.S[:, 0] += - self.border_condition_values[1] * self.Ny

        if self.border_condition_types[3] == 'dir':
            self.S[:, -1] += self.border_condition_values[3] * self.Ny**2
        else:
            L2[-1, -2] += 1 * self.Ny ** 2
            self.S[:, -1] += self.border_condition_values[3] * self.Ny


        self.MAT_DELTA = sps.kron(sps.eye(self.Nx), L2) + sps.kron(L1, sps.eye(self.Ny))
        self.MAT1 = self.MAT_DELTA


        self.diag = np.ravel(np.sqrt((dx(self.PHI) * self.Nx) ** 2 + (dx(self.PHI.T).T * self.Ny) ** 2))

        self.MAT2 = sps.diags([self.diag], [0], dtype=float, format="csr")

        self.S = np.ravel(self.S)



    def tick(self, dt):
        for i in range(100):
            newU = (self.U + dt * (np.reshape(self.MAT1.dot(np.ravel(self.U)) + self.S, (self.Nx, self.Ny)) - self.LAMBDA * self.D))/(1 + dt * self.s * self.D ** 2)
            self.LAMBDA += self.s * self.D * newU

        self.U = np.copy(newU)


    def getU(self):
        return self.U