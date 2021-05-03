#@title Solving ODE
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline
import scipy.special as sc
import matplotlib.pyplot as plt
from george.modeling import Model

#---------------------------
def system_ODE_GP(vec,a,p,zp,mu,om):
  #LCDM currently working!
    #Hs=lambda a: np.sqrt(om *a**(-3) + (1-om))
    #Hsp=lambda a:(-3*om*a**(-4))/(2*np.sqrt(1-om +om*a**(-3)))
    Hsz=UnivariateSpline(zp,p,k=3,s=0.01)
    Hspz=Hsz.derivative()
    Hs=lambda a: Hsz(1/a-1)
    Hsp=lambda a: -1/a**2 * Hspz(1/a - 1)

    #System ODE
    x1,x2=vec[0],vec[1]
    x1p=x2
    #x2p=3/2*(a**(-5)*om)/(Hs(a))**2 *x1 - (3/a + (Hsp(a)/Hs(a)))*x2
    x2p=3/2*(a**(-5)*om*mu.get_value(1/a - 1))/Hs(a)**2 *x1 - (3/a + (Hsp(a)/Hs(a)))*x2
    return np.array([x1p,x2p])

#---------------------------
def get_sols_a(a,p,zp,mu,om=0.3,s8=0.81):
  ''' --------------
   Returns delta_m(a), f(a) and fs8(a). Must input cosmological parameters:

   - p = H(a)/H0 - array with (normalized) prediction from GP,
   - mu(a),
   - Omega_m0 ,
   - s8,0

  in that order.
  ------------  '''

  # ODE solver parameters
  abserr = 1.0e-8
  relerr = 1.0e-8

  aini=a[0]
  y0=[aini,1]
  sols=odeint(system_ODE_GP,y0,a,args=(p,zp,mu,om,), atol=abserr, rtol=relerr,h0=10**(-10))
  d,dp=sols[:,0], sols[:,1]
  d,f,fs8 = d, a/d*dp, s8*a/(d[-1])*dp
  return (d, f, fs8)

#---------------------------
class g_fR(Model):
    parameter_names = ("amp","loc")

    def get_value(self, z):
        #z = z.flatten()
        return (1+self.amp*np.exp(-1*(z-self.loc)**2))

class g_ST(Model):
    parameter_names = ("amp",'loc')

    def get_value(self, z):
        #z = z.flatten()
        return ( 1+self.amp + (self.amp * np.tanh(15*(z-self.loc))) )

class g_ST2(Model):
    parameter_names = ("amp",'loc')

    def get_value(self, z):
        #z = z.flatten()
        return ( 1+self.amp - (self.amp * np.tanh(15*(z-self.loc))) )
#---------------------------

#---------------------------
def analytical_delta(a,w,om):
  #a=1./(1+zz)
  b, c, d, e=  -1/(3*w), 0.5 - 1/(2*w), 1 - 5/(6*w), (1 - 1/om)*a**(-3*w)
  #D_today = sc.hyp2f1(b,c,d,(1-1/om))
  return a * sc.hyp2f1(b, c, d, e)#/D_today

#---------------------------
# Approximation growth function \Om^gamma (z)

def f_approx(z,gamma=0.55,om=0.3):
  return (om/(om*(1+z)**3 +(1-om))*(1+z)**3)**gamma

def fs8_approx(a,om=0.3,g=0.55,s8=0.81):
  return f_approx(1/a-1,g,om)*s8*analytical_delta(a,-1,om)/analytical_delta(1,-1,om)
