r"""
One dimension shallow water, dam break case.
============================================

This validation example use the shallow water equation to
model a dam sudden break. The two part of the domain have different
fluid depth and are separated by a well. A a certain instant, the wall
disappear, leading to a discontinuity wave in direction of the lower depth,
and a rarefaction wave in direction of the higher depth.

The model reads as

.. math::

    \begin{cases}
        \frac{\partial h}{\partial t} + \frac{\partial h}{\partial x} &= 0 \\
        \frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} &= -g\,\frac{\partial h}{\partial x}
    \end{cases}

and the results are validated on the Randall J. LeVeque book (LeVeque, R. (2002).
Finite Volume Methods for Hyperbolic Problems (Cambridge Texts in Applied Mathematics).
Cambridge: Cambridge University Press. doi:10.1017/CBO9780511791253).
"""

import pylab as pl
import pandas as pd
from skfdiff import Model, Simulation
import numpy as np
from scipy.signal import savgol_filter
import csv

eqn1 = "-dxq"
eqn22 = "-1.0*Q0/(epsilon*re*Q1) - Q1_tilde/Q1 - epsilon*Q2/(re*Q1)"
eqn21 = "-1.0*Q0/(epsilon*re*Q1) - Q1_tilde/Q1"

Q0_exp = "((h-1.0)*(h**2.0*G*(rho-1.0) + a1)-(m*b1*h**2)/(h-1.0))*(h*(m-1.0)+1.0)"
Q1_exp = "(1/20.0)*(h**2.0)*(8*h**2.0*(m-1) + 16*h - 7*m*h - 8)*a1q + (1/8)*(h**2.0)*(5*h**2*(m-1) - 4*m*h + 10*h - 5)*a0q - (1/8)*h*rho*(h-1)*(5*h**2*(m-1) + 6*h - 1)*b0q - (1/20)*h*rho*(h-1)*(8*h**2*(m-1) + 9*h -1)*b1q"
Q1_tilde_exp = "(1/re)*h**2*(1-h)*(h*(m-1)+1)*((rho-1)*G*(1/So)*dxh + epsilon**2*we*dxxxh) + (1/120)*h*a0*(h*a1x-2*dxh*a1)*(13*h**2*(m-1)-10*m*h+26*h-13) - (3/(40*h*m**2))*rho*(h-1)*(3*h**2*(m-1)+4*h-1)*(m*h*b0x*(m*h*b1+a0*(h-1))-dxh*a0**2*(h-1)) - (1/840)*h*dxh*(a1**2)*(33*h**2*(m-1)-34*m*h+66*h-33) + (1/840)*rho*h*dxh*(b1**2)*(33*h**2*(m-1)+32*h+1) + (3/40)*h*(h*a0x*a0+h*a0x*a1-dxh*a0**2)*(3*h**2*(m-1)-2*m*h+6*h-3) + (1/(60*m))*rho*(h-1)*a0*(13*h**2*(m-1)+16*h-3)*(dxh*b1-(1/2)*(h-1)*b1x) - (1/840)*rho*h*(h-1)*(111*h**2*b1*(m-1)+142*b1h-31*b1)*b1x + (1/840)*h**2*(a1*a1x)*(111*h**2*(m-1)+222*h-80*m*h-111) + dxq*((-1/40)*h*rho*(7*h**2*(m-1)+6*h+1)*b1+(1/20)*h*rho*(h-1)*b1h*(8*h**2*(m-1)+9*h-1)-(1/20)*h**2*a1h*(8*h**2*(m-1)+16*h-7*m*h-8)+(1/40)*h*a1*(7*h**2*(m-1)-8*m*h+14*h-7)+(1/(8*m)*rho*(h-1)*(5*h**2*(m-1)+6*h-1)*(m*h*b0h-a0))-(1/8)*h*(5*h**2*(m-1)-4*m*h+10*h-5)*(a0h*h-a0))"
Q2_exp = "(3*h**2*(m-1)+4*h-1)*(3/5*h*m*b1xx*(h-1)+m*h*b0xx*(h-1)-1/5*m*h*b1*dxxh-a0*h*dxxh+a0*dxxh) - (1/5*h*(3*h**2*(m-1)-2*m*h+6*h-3)*(3*h*a1xx+5*h*a0xx-5*a0*dxxh-a1*dxxh)) - dxh*(h**2*(m-1)+3*m*h+2*h-1)*(a0x*h-a0*dxh) + (1/5)*(h**2*(m-1)+2*h+m*h-1)*a1*(dxh**2) + dxh*(h**2*(m-1)+5*h-4)*(b0x*m*h-a0*dxh) - (1/5*h*dxh*a1x)*(9*h*(m-1)+4*m*h+18*h-9) + (1/5*h*m/(1-h)*b1*(dxh**2)*(h**2*(m-1)-2-h**2)) + (1/5*h*m*dxh*b1x*(9*h**2*(m-1)+22*h-13))"

a1_exp = "((-1)+h)**(-1)*h**(-1)*(1+((-1)+m)*h)**(-1)*(9*m*h**2+3*((-1)+h*(2+(-4)*m+((-1)+m)*h))*q)"
b1_exp = "((-1)+h)**(-1)*h**(-1)*(1+((-1)+m)*h)**(-1)*(9*q+(-6)*h*(2+q)+3*h**2*(4+(-1)*m+((-1)+m)*q))"
a0_exp = "6*m*((-1)+h)**(-1)*(1+((-1)+m)*h)**(-1)*((-1)*h+q)"
a1q_exp = "3*h**(-1)+(-9)*m*((-1)+h)**(-1)*(1+((-1)+m)*h)**(-1)"
a0q_exp = "6*m*((-1)+h)**(-1)*(1+((-1)+m)*h)**(-1)"
b1q_exp = "3*((-1)+h)**(-1)+(-9)*h**(-1)+9*((-1)+m)*(1+((-1)+m)*h)**(-1)"
b0q_exp = "6*(h+((-1)+m)*h**2)**(-1)"
a0h_exp = "6*m*((-1)+h)**(-2)*(1+((-1)+m)*h)**(-2)*(1+((-1)+m)*h**2+((-2)+m)*q+(-2)*((-1)+m)*h*q)"
a1h_exp = "(-1)*((-1)+h)**(-2)*h**(-2)*(1+((-1)+m)*h)**(-2)*(9*m*h**2+9*((-1)+m)*m*h**4+3*(1+h*(2*((-2)+m)+h*(6+4*((-3)+m)*m+((-1)+m)*h*(4+(-8)*m+((-1)+m)*h))))*q)"
b0h_exp = "(h+((-1)+m)*h**2)**(-2)*(6*((-1)+m)*h*(h+(-2)*q)+(-6)*q)"
b1h_exp = "((-1)+h)**(-2)*h**(-2)*(1+((-1)+m)*h)**(-2)*(9*q+3*h*(6*((-2)+m)*q+4*((-1)+m)*h**2*(2+q)+h*(4+(-3)*m+2*(7+(-6)*m)*q)+(-1)*((-1)+m)*h**3*(4+(-1)*m+((-1)+m)*q)))"
a0x_exp = "6*m*((-1)+h)**(-2)*(1+((-1)+m)*h)**(-2)*((1+((-1)+m)*h**2+((-2)+m)*q+(-2)*((-1)+m)*h*q)*dxh+((-1)+h)*(1+((-1)+m)*h)*dxq)"
a1x_exp = "((-1)+h)**(-2)*h**(-2)*(1+((-1)+m)*h)**(-2)*((-3)*(3*m*h**2*(1+((-1)+m)*h**2)+(1+h*(2*((-2)+m)+h*(6+4*((-3)+m)*m+((-1)+m)*h*(4+(-8)*m+((-1)+m)*h))))*q)*dxh+3*((-1)+h)*h*(1+((-1)+m)*h)*((-1)+h*(2+(-4)*m+((-1)+m)*h))*dxq)"
b0x_exp = "(h+((-1)+m)*h**2)**(-2)*(6*(((-1)+m)*h*(h+(-2)*q)+(-1)*q)*dxh+6*h*(1+((-1)+m)*h)*dxq)"
b1x_exp = "3*((-1)+h)**(-2)*h**(-2)*(1+((-1)+m)*h)**(-2)*(3*q*dxh+3*h*(2*((-2)+m)*q*dxh+(-1)*dxq)+((-1)+m)**2*h**5*dxq+h**2*((4+(-3)*m+2*(7+(-6)*m)*q)*dxh+(8+(-3)*m)*dxq)+((-1)+m)*h**4*(((-4)+m+q+(-1)*m*q)*dxh+(-1)*m*dxq)+2*h**3*(2*((-1)+m)*(2+q)*dxh+((-3)+2*m)*dxq))"
a0xx_exp = "6*m*((-1)+h)**(-3)*(1+((-1)+m)*h)**(-3)*((-2)*(2+(-1)*m+((-1)+m)**2*h**3+(-1)*(3+((-3)+m)*m)*q+(-3)*((-1)+m)**2*h**2*q+3*((-1)+m)*h*(1+((-2)+m)*q))*dxh**2+(-2)*((-1)+h)*(1+((-1)+m)*h)*(2+(-1)*m+2*((-1)+m)*h)*dxh*dxq+((-1)+h)*(1+((-1)+m)*h)*((1+((-1)+m)*h**2+((-2)+m)*q+(-2)*((-1)+m)*h*q)*dxxh+((-1)+h)*(1+((-1)+m)*h)*dxxq))"
a1xx_exp = "3*((-1)+h)**(-3)*h**(-3)*(1+((-1)+m)*h)**(-3)*((-2)*q*dxh**2+h*(2*dxh*dxq+q*((-6)*((-2)+m)*dxh**2+dxxh))+((-1)+m)**3*h**8*dxxq+(-1)*h**2*((-6)*((-2)+m)*dxh*dxq+3*q*(2*(5+((-5)+m)*m)*dxh**2+(-1)*((-2)+m)*dxxh)+dxxq)+h**3*(((-6)*((-2)+m)*m+2*(20+m*((-39)+(21+(-4)*m)*m))*q)*dxh**2+6*((-1)+m)*((-5)+2*m)*dxh*dxq+3*(m+((-1)+m)*((-5)+2*m)*q)*dxxh+(-6)*((-1)+m)*dxxq)+(-1)*((-1)+m)**2*h**7*(2*((-1)+m)*dxh*dxq+(3*m+((-1)+m)*q)*dxxh+6*((-1)+m)*dxxq)+h**4*(6*((-1)+m)*(3*m+(5+m*((-11)+4*m))*q)*dxh**2+4*((-10)+m*(24+m*((-15)+2*m)))*dxh*dxq+(3*((-2)+m)*m+2*((-10)+m*(24+m*((-15)+2*m)))*q)*dxxh+(-3)*(5+3*((-3)+m)*m)*dxxq)+((-1)+m)*h**6*(2*((-1)+m)*(3*m+((-1)+m)*q)*dxh**2+6*((-1)+m)*((-2)+3*m)*dxh*dxq+3*(((-2)+m)*m+((-1)+m)*((-2)+3*m)*q)*dxxh+3*(5+3*((-3)+m)*m)*dxxq)+h**5*(3*((-1)+m)*((-2)*(5+m*((-11)+4*m))*dxh*dxq+q*((-4)*((-1)+m)*((-1)+2*m)*dxh**2+((-5)+(11+(-4)*m)*m)*dxxh))+2*(10+m*((-24)+(15+(-2)*m)*m))*dxxq))"
b0xx_exp = "(h+((-1)+m)*h**2)**(-3)*(2*(1+2*((-1)+m)*h)**2*((-6)*h+6*q)*dxh**2+(-2)*(1+2*((-1)+m)*h)*(h+((-1)+m)*h**2)*dxh*((-6)*dxh+6*dxq)+(-1)*(h+((-1)+m)*h**2)*((-6)*h+6*q)*(2*((-1)+m)*dxh**2+(1+2*((-1)+m)*h)*dxxh)+(h+((-1)+m)*h**2)**2*((-6)*dxxh+6*dxxq))"
b1xx_exp = "3*((-1)+h)**(-3)*h**(-3)*(1+((-1)+m)*h)**(-3)*(6*q*dxh**2+3*h*((-2)*dxh*dxq+q*(6*((-2)+m)*dxh**2+(-1)*dxxh))+((-1)+m)**3*h**8*dxxq+3*h**2*((-6)*((-2)+m)*dxh*dxq+3*q*(2*(5+((-5)+m)*m)*dxh**2+(-1)*((-2)+m)*dxxh)+dxxq)+(-1)*h**3*(2*(4+3*((-2)+m)*m+(56+27*((-3)+m)*m)*q)*dxh**2+2*(41+(-39)*m+6*m**2)*dxh*dxq+(4+(-3)*m+(41+(-39)*m+6*m**2)*q)*dxxh+2*(7+(-3)*m)*dxxq)+(-1)*((-1)+m)**2*h**7*(2*((-1)+m)*dxh*dxq+(4+(-1)*m+((-1)+m)*q)*dxxh+2*((-1)+m)*dxxq)+h**4*(6*((-1)+m)*((-4)+3*m+((-11)+9*m)*q)*dxh**2+4*(22+(-30)*m+9*m**2)*dxh*dxq+(16+3*((-6)+m)*m+2*(22+(-30)*m+9*m**2)*q)*dxxh+(25+3*((-7)+m)*m)*dxxq)+h**5*(3*((-1)+m)*((-4)*((-1)+m)*(2+q)*dxh**2+2*(7+(-5)*m)*dxh*dxq+(8+(-4)*m+(7+(-5)*m)*q)*dxxh)+(-2)*(10+3*((-4)+m)*m)*dxxq)+((-1)+m)*h**6*(2*((-1)+m)*(4+(-1)*m+((-1)+m)*q)*dxh**2+2*((-2)+m+m**2)*dxh*dxq+((-16)+(-1)*((-14)+m)*m+((-2)+m+m**2)*q)*dxxh+((-5)+m+m**2)*dxxq))"

bc = {("h", "x"): ("periodic", "periodic"),
      ("q", "x"): ("periodic", "periodic")}

#shallow_water = Model([eqn1, eqn21], ["h(x)", "q(x)"], parameters=["re", "we", "epsilon", "So", "G", "m", "rho"], subs=dict(Q0=Q0_exp, Q1=Q1_exp, Q1_tilde=Q1_tilde_exp, Q2=Q2_exp, a1=a1_exp, b1=b1_exp, a0=a0_exp, a1q=a1q_exp, a0q=a0q_exp, b1q=b1q_exp, b0q=b0q_exp, a0h=a0h_exp, a1h=a1h_exp, b0h=b0h_exp, b1h=b1h_exp, a0x=a0x_exp, a1x=a1x_exp, b0x=b0x_exp, b1x=b1x_exp, a0xx=a0xx_exp, a1xx=a1xx_exp, b0xx=b0xx_exp, b1xx=b1xx_exp), boundary_conditions=bc)
shallow_water = Model([eqn1, eqn21], ["h(x)", "q(x)"], parameters=["re", "we", "epsilon", "So", "G", "m", "rho"], subs=dict(Q0=Q0_exp, Q1=Q1_exp, Q1_tilde=Q1_tilde_exp, Q2=Q2_exp, a1=a1_exp, b1=b1_exp, a0=a0_exp, a1q=a1q_exp, a0q=a0q_exp, b1q=b1q_exp, b0q=b0q_exp, a0h=a0h_exp, a1h=a1h_exp, b0h=b0h_exp, b1h=b1h_exp, a0x=a0x_exp, a1x=a1x_exp, b0x=b0x_exp, b1x=b1x_exp, a0xx=a0xx_exp, a1xx=a1xx_exp, b0xx=b0xx_exp, b1xx=b1xx_exp), boundary_conditions="periodic")

###############################################################################
# Filter post-process
# -------------------
#
# As the discontinuity will be still harsh to handle by a "simple" finite
# difference solver (a finite volume solver is usually the tool you will need),
# we will use a gaussian filter that will be applied to the fluid height. This
# will smooth the oscillation (generated by numerical errors). This can be seen
# as a way to introduce numerical diffusion to the system, often done by adding
# a diffusion term in the model. The filter has to be carefully tuned (the same
# way an artificial diffusion has its coefficient diffusion carefully chosen)
# to smooth the numerical oscillation without affecting the physical behavior
# of the simulation.


def filter_instabilities(simul):
    simul.fields["h"] = ("x",), savgol_filter(simul.fields["h"], 8, 4)
    simul.fields["q"] = ("x",), savgol_filter(simul.fields["q"], 8, 4)

def hook(t, fields):
    #l_x = 5.0
    #N_x = 200
    format_string_time = f"{t:.2f}"
    file_name = 'outXYZ_%s' % format_string_time
    # We bound the value of T, to be sure it does not go over 0.5
    if (np.mod(t, t_output)<=dt_val):
        h = fields["h"].values
        q = fields["q"].values
        x_arr, dx_val = np.linspace(0, l_x, N_x, retstep=True)
        with open(file_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(np.transpose(x_arr), np.transpose(h), np.transpose(q)))

    # max depth tracking
    if (np.mod(t, (dt_val*15.0))<=dt_val):
        h = fields["h"].values
        q = fields["q"].values
        max_h = np.max(h)
        max_h_ind = np.argmax(h)
        count = max_h_ind
        max_u = q[count]/h[count]
        f1 = open("maxHU.txt", "a+")
        f1.write("%g %g %18.8e %18.8e \n" % (t, ((l_x/N_x)*(max_h_ind-0.50)), max_h, max_u))

    return fields

##### TL modified here #####
##### problem-specific parameters (dimensionless) #####
##### ρ = 0.5, m = 0.25, h1 = 0.5, Fr = 0.80, So = 0.05, no Surface Tension #####
So_val = 0.050;
l_x = 5.0
N_x = 500
h1_val = 0.40;
m_val = 0.25;
rho_val = 0.50;
fr_val = 0.80;
we_val = 0.00;

r = (1.0-h1_val)/h1_val
beta = (m_val+r)**(-1.0)*(1.0+r*rho_val)

re_val = (fr_val**2.0/So_val)/(((r*(1+r)*(1+r*rho_val))/(m_val+r)+(m_val+r**3.0*rho_val)/(3.0*m_val))/(4*(1+r)**3.0)) # the "R" param_valeter #
G_val = 4*(1+r)**3*(r*(1+r)*(m_val+r)**(-1)*(1+r*rho_val)+(1/3)*m_val**(-1)*(m_val+r**3*rho_val))**(-1) # the "G" param_valeter #

epsilon_val = So_val

dist_amp = 0.10
t_output = 0.50
dt_val = 0.010
tf = 69.10

x, dx = np.linspace(0, l_x, N_x, retstep=True)
h = (1.0)/(1.0+r)*(1.0+dist_amp*np.sin(2*np.pi*x/l_x))
q = m_val*(m_val+r*(4+3*r*rho_val))*(m_val**2+r**4*rho_val+m_val*r*(4+3*r+r*(3+4*r)*rho_val))**(-1)

init_fields = shallow_water.Fields(x=x, h=h, q=q, re=re_val, we=0.0, So=So_val, epsilon=So_val, G=G_val, m=m_val, rho=rho_val)

simul = Simulation(
    shallow_water,
    t=0,
    dt=dt_val,
    tmax=tf,
    #scheme="Theta",  # using a theta scheme
    #theta=0.5,  # use theta=0.5 for the temporal scheme
    #scheme="scipy_ivp",
    #method="RK45",
    #method="RK23",
    fields=init_fields,
    time_stepping=True, # allow a variable internal time-step to ensure good agreement between computing performance and accuracy.
    id="periodic",
    hook=hook,
    backend="numba",
)

simul.add_post_process("filter", filter_instabilities)

container = simul.attach_container()

simul.run()

