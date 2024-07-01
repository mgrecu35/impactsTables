import torch
from math import cos

def tbf90_py(I01p, T, incAng, k, a, g, n, eps, dz):
    pi = 3.1415
    cos1 = cos(incAng / 180.0 * pi)
    
    Jup = torch.zeros(n + 1)
    Jd = torch.zeros(n + 1)
    
    for k1 in range(n + 1):
        Jup[k1] = (1 - a[k1]) * T[k1] + a[k1] * (0.5 * (I01p[2*k1] + I01p[2*k1 + 2]) * g[k1] * cos1 + I01p[2*k1 + 1])
        Jd[k1] = (1 - a[k1]) * T[k1] + a[k1] * (-0.5 * (I01p[2*k1] + I01p[2*k1 + 2]) * g[k1] * cos1 + I01p[2*k1 + 1])
    
    intDo = torch.zeros(n + 1)
    intUp = torch.zeros(n + 1)
    
    intDo[0] = k[0] * dz / 2
    intUp[n] = k[n] * dz / 2
    
    for k1 in range(1, n + 1):
        intUp[n - k1] = intUp[n - k1 + 1] + 0.5 * (k[n - k1] + k[n - k1 + 1]) * dz
    
    for k1 in range(1, n + 1):
        intDo[k1] = intDo[k1 - 1] + 0.5 * (k[k1] + k[k1 - 1]) * dz
    
    sumJD = torch.tensor(0.0)
    sumJU = torch.tensor(0.0)
    
    for i in range(n + 1):
        sumJD += Jd[i] * torch.exp(-intDo[i] / cos1) * k[i] * dz / cos1
        sumJU += Jup[i] * torch.exp(-intUp[i] / cos1) * k[i] * dz / cos1
    
    Tb = (1 - eps) * torch.exp(-intUp[0] / cos1) * sumJD + sumJU + eps * torch.exp(-intUp[0] / cos1) * T[0]
    return Tb


def SetEddington1D(k, a, g, T, n, eps, dz, Ts):
    Abig = torch.zeros((2*n+3, 2*n+3), dtype=torch.float32)
    B = torch.zeros(2*n+3, dtype=torch.float32)
    
    for i in range(n + 1):
        Abig[2*i + 1, 2*i + 1] = 3 * k[i] * (1 - a[i]) * dz
        Abig[2*i + 1, 2*i + 2] = 1
        Abig[2*i + 1, 2*i] = -1
        B[2*i + 1] = 3 * k[i] * (1 - a[i]) * T[i] * dz
        
        if 2*i + 2 < 2*n + 3 - 2:
            km = 0.5 * (k[i] + k[i + 1])
            am = 0.5 * (a[i] + a[i + 1])
            gm = 0.5 * (g[i] + g[i + 1])
            Abig[2*i + 2, 2*i + 2] = km * (1 - am * gm) * dz
            Abig[2*i + 2, 2*i + 3] = 1
            Abig[2*i + 2, 2*i + 1] = -1
            B[2*i + 2] = 0.0
    
    Abig[2*n + 2, 2*n + 1] = 1.0
    Abig[2*n + 2, 2*n + 2] = -1.0 / 3
    Abig[2*n + 2, 2*n] = -1.0 / 3
    B[2*n + 2] = 2.7
    
    Abig[0, 0] = (2 - eps) / (3 * eps)
    Abig[0, 1] = 1.0
    Abig[0, 2] = (2 - eps) / (3 * eps)
    B[0] = Ts
    return Abig, B

def tbwrapper(kext1, scat1, asym1, temp_mid, dz, eps, n, incAng):
    #temp_mid = temp1[1:-1]
    Abig_py, B_py = SetEddington1D(kext1, scat1, asym1, temp_mid, n, eps, dz, temp_mid[0])
    I01p_py = torch.linalg.solve(Abig_py, B_py)
    Tb_py = tbf90_py(I01p_py, temp_mid, incAng, kext1, scat1, asym1, n, eps, dz)
    return Tb_py


# Example usage:
# Define your input tensors with requires_grad=True if you want to calculate gradients with respect to them
def torch_grad(kext, scat, asym, temp):
    kext1 = kext[:].copy()
    scat1 = scat[:].copy()
    asym1 = asym[:].copy()
    temp1 = temp[:].copy()
    temp_mid = (temp1[:-1] + temp1[1:]) / 2
    kext1_torch = torch.tensor(kext1, dtype=torch.float32, requires_grad=True)
    scat1_torch = torch.tensor(scat1, dtype=torch.float32, requires_grad=True)
    asym1_torch = torch.tensor(asym1, dtype=torch.float32, requires_grad=True)
    temp_mid_torch = torch.tensor(temp_mid, dtype=torch.float32, requires_grad=True)
    #print(kext1.shape, scat1.shape, asym1.shape, temp_mid.shape)
# dz, eps, n, and incAng can remain as Python variables
    dz = 0.25  # example value
    eps = torch.tensor(0.75, dtype=torch.float32, requires_grad=True)  # example value
    n = 79    # example value
    incAng = 53.0  # example value
    Tb_torch = tbwrapper(kext1_torch, scat1_torch, asym1_torch, temp_mid_torch, dz, eps, n, incAng)
    Tb_torch.backward()
    return kext1_torch.grad.detach().numpy(), scat1_torch.grad.detach().numpy(), asym1_torch.grad.detach().numpy(), temp_mid_torch.grad.detach().numpy()