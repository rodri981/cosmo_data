#@title Solving ODE
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import quad
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
#   d,dp=sols[:,0], sols[:,1]
#   d,f,fs8 = sols[:,0], a/sols[:,0]*sols[:,1], s8*a/(sols[:,0][-1])*sols[:,1]
  return (sols[:,0], a/sols[:,0]*sols[:,1], s8*a/(sols[:,0][-1])*sols[:,1])



def system_ODE_CPL(vec,a,p,mu):
    om,w0,wa=p
    Hs=lambda a: np.sqrt((np.exp(-3*wa + 3*a*wa)*(1 - om))/a**(3*(1 + w0 + wa)) + om/a**3)
    Hsp=lambda a: ((-3*om)/a**4 + (3*np.exp(-3*wa + 3*a*wa)*(1 - om)*wa)/a**(3*(1 + w0 + wa))-3*a**(-1-3*(1 + w0 + wa))*np.exp(-3*wa+3*a*wa)*(1-om)*(1 + w0 + wa))/(2.*np.sqrt((np.exp(-3*wa + 3*a*wa)*(1 - om))/a**(3*(1+w0+wa))+om/a**3))

    #System ODE
    x1,x2=vec[0],vec[1]
    x1p=x2
    #x2p=3/2*(a**(-5)*om)/(Hs(a))**2 *x1 - (3/a + (Hsp(a)/Hs(a)))*x2
    x2p=3/2*(a**(-5)*om*mu.get_value(1/a-1))/Hs(a)**2 *x1 - (3/a + (Hsp(a)/Hs(a)))*x2
    return np.array([x1p,x2p])

#---------------------------
def get_sols_CPL(a,p,mu,s8):
  ''' --------------
   Returns delta_m(a), f(a) and fs8(a). Must input cosmological parameters:

   - p = (om,w0,wa) - tuple with cosmological parameters to instantiate the CPL class,
   - mu(a),
   - s8,0

  in that order.
  ------------  '''

  # ODE solver parameters
  abserr = 1.0e-10
  relerr = 1.0e-10

  aini=a[0]
  y0=[aini,1]
  sols=odeint(system_ODE_CPL,y0,a,args=(p,mu,), atol=abserr, rtol=relerr,h0=10**(-10))
  d,f,fs8 = sols[:,0], a/sols[:,0]*sols[:,1], s8*a/(sols[:,0][-1])*sols[:,1]
  return (d, f, fs8)

#---------------------------
def rhode(a,w):
  # from scipy.integrate import quad

  #this gives a callable for (1+w(a))/a needed for quad integration
  onepw_integrand=UnivariateSpline(a_pred,(1+w)/a_pred.flatten(),k=3,s=0.1)
  if isiterable(a):
      a = np.asarray(a)
      ival = np.array([quad(onepw_integrand, aa, 1)[0]
                        for aa in a])
      return np.exp(3 * ival)
  else:
      ival = quad(onepw_integrand,a, 1)[0]
      return np.exp(3 * ival)

#---------------------------
def system_ODE_w(vec,a,wgp,a_pred,mu,om):
    onepw=UnivariateSpline(a_pred,(1+wgp),k=3,s=0.1)
    Hs=lambda a: np.sqrt(om*a**(-3)+(1-om)*rhode(a,wgp))
    Hsp=lambda a: -3/2*(om*a**(-4)+rhode(a,wgp)*(1-om)*onepw(a)/a)/np.sqrt(om*a**(-3)+(1-om)*rhode(a,wgp))

    #System ODE
    x1,x2=vec[0],vec[1]
    x1p=x2
    x2p=3/2*(a**(-5)*om*mu.get_value(1/a - 1))/Hs(a)**2 *x1 - (3/a + (Hsp(a)/Hs(a)))*x2
    return np.array([x1p,x2p])

#---------------------------
def get_sols_w(a,wgp,ap,om,mu,s8):
  ''' --------------
   Returns delta_m(a), f(a) and fs8(a). Must input the following:

   - wgp = w(ap)  - array with w sample from GP on ap=a_pred,
   - ap - array where the prediction was computed
   - Omega_m0 ,
   - mu(a),
   - s8,0

  in that order.
  ------------  '''

  # ODE solver parameters
  abserr = 1.0e-10
  relerr = 1.0e-10

  aini=a[0]
  y0=[aini,1]
  sols=odeint(system_ODE_w,y0,a,args=(wgp,ap,mu,om,), atol=abserr, rtol=relerr,h0=10**(-10))
  d,f,fs8 = sols[:,0], a/sols[:,0]*sols[:,1], s8*a/(sols[:,0][-1])*sols[:,1]
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
