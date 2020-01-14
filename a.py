import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

cuda = torch.device('cuda')


size = 4096
delta = 0.1
dL = 0.01
c = 3e8
Hz = 0.4e12

"""
def propogation(u0, d=delta, N = size, dL = dL, lmb = 682e-9):
    #Parameter 
    df = 1.0/dL
    k = np.pi*2.0/lmb
    D= dL*dL/(N*lmb)
  
    #phase
    def phase(i,j):
        i -= N//2
        j -= N//2
        return ((i*df)*(i*df)+(j*df)*(j*df))
    ph  = np.fromfunction(phase,shape=(N,N),dtype=np.float32)
    #H
    H = np.exp(1.0j*k*d)*np.exp(-1.0j*lmb*np.pi*d*ph) 
    #Result
    return np.fft.ifft2(np.fft.fftshift(H)*np.fft.fft2(u0)*dL*dL/(N*N))*N*N*df*df
"""

def propogation_(u0, d=delta, N = size, dL = dL, lmb = 682e-9):
    #Parameter 
    df = 1.0/dL
    k = np.pi*2.0/lmb
    D= dL*dL/(N*lmb)
  
    #phase
    def phase(i,j):
        i -= N//2
        j -= N//2
        return ((i*df)*(i*df)+(j*df)*(j*df))
    ph  = np.fromfunction(phase,shape=(N,N),dtype=np.float32)
    #H
    H = np.exp(1.0j*k*d)*np.exp(-1.0j*lmb*np.pi*d*ph)
    z = np.fft.fftshift(H)
    zz = torch.zeros(N,N,2)
    zz.numpy()[:,:,0] = np.real(z)
    zz.numpy()[:,:,1] = np.imag(z) 
    u = torch.zeros(N,N,2)
    u.numpy()[:,:,0] = u0
    fft_mat = torch.fft(u,2,normalized=True)

    #Result
    return torch.ifft(zz*fft_mat*dL*dL/(N*N), 2, normalized=True)*N*N*df*df


lmb = 682e-9
d = delta
N = size
dL = dL
df = 1.0/dL
k = np.pi*2.0/lmb
D= dL*dL/(N*lmb)

def phase(i,j):
        i -= N//2
        j -= N//2
        return ((i*df)*(i*df)+(j*df)*(j*df))
phh  = np.fromfunction(phase,shape=(N,N),dtype=np.float32)
    #H
H = np.exp(1.0j*k*d)*np.exp(-1.0j*lmb*np.pi*d*phh)
z = np.fft.fftshift(H)
zz = torch.zeros(N,N,2)
zz.numpy()[:,:,0] = np.real(z)
zz.numpy()[:,:,1] = np.imag(z) 

def prop(u0, d=delta, N = size, dL = dL, lmb = 682e-9, zz=zz):
    
    df = 1.0/dL
    k = np.pi*2.0/lmb
    D= dL*dL/(N*lmb)

    fft_mat = torch.fft(u0,2,normalized=True)

    return torch.ifft(zz.cuda()*fft_mat*dL*dL/(N*N), 2, normalized=True)*N*N*df*df


def norm(i,j):
    n = size//2
    return ((i-n)*(i-n)+(j-n)*(j-n))
n = np.fromfunction(norm,shape=(size,size),dtype=np.float32)
n = torch.from_numpy(n)

def cost(u, N = size, norm=n):
    u0 = u[:,:,0]**2+u[:,:,1]**2
    return torch.sum(u0*norm.cuda())




x = np.ones((4096,4096))
hole = cv2.circle(x, (2048,2048),800,0,-1)
"""
vertices = np.array([[2000,1500],[1500,2500],[2500,2500]])
pts = vertices.reshape((-1,1,2))
hole = cv2.fillPoly(x, [pts], color=1)
"""
#plt.imshow(hole)
plt.imshow(np.abs(propogation(hole))**2)


ph = torch.randn(4096,4096,requires_grad=True, device=cuda)
rate = 1
optimizer = torch.optim.Adam([ph],lr=rate)
l = []
for i in range(1000):
    optimizer.zero_grad()
    amp = torch.stack((torch.cos(ph),torch.sin(ph)), dim=2).cuda()
    c = cost(prop(amp))
    c.backward()
    l.append(int(c.cpu().detach().numpy()))
    optimizer.step()

amp = torch.stack((torch.cos(ph),torch.sin(ph)), dim=2)
z = prop(amp)
plt.imshow((z[:,:,0]**2+z[:,:,1]**2).detach())

np.mod(ph.detach().numpy(),2.0*np.pi)
