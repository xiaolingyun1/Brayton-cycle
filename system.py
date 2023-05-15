import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
from scipy.linalg import solve
import pandas as pd
from scipy.integrate import quad
import matplotlib
import warnings
warnings.simplefilter('always', UserWarning)
matplotlib.use('Agg')

# adiabatic process
def ad(p1,h,p2, fluid):
    s = cp.PropsSI('S', 'P', p1, 'H', h, fluid)
    h2 = cp.PropsSI('H', 'P', p2, 'S', s, fluid)
    return h2
# Define function to plot T-s diagram
def plot_TS_diagram(fluidname, Tmin, Tmax, ax):
    t = np.linspace(Tmin, Tmax, 200)
    sL = cp.PropsSI('S', 'T', t, 'Q', 0, fluidname)
    sV = cp.PropsSI('S', 'T', t, 'Q', 1, fluidname)
    ax.plot(sL, t, 'b-', label='Saturated Liquid')
    ax.plot(sV, t, 'r-', label='Saturated Vapor')

# Define function to plot isobaric process on T-s diagram
def plot_isobar_TS_diagram(fluidname, p, Tmin, Tmax, ax,label=''):
    T = np.linspace(Tmin, Tmax, 101)
    s = cp.PropsSI('S', 'T', T, 'P', p, fluidname)
    ax.plot(s, T, 'g-',label=label)

# Plot the isothermal process on the T-s diagram
def plot_isothermal_TS_diagram(s_min, s_max, Tmin, Tmax,ax,label=''):
    ax.plot([s_min, s_max], [Tmin, Tmax], 'k-',label=label)

# T = np.zeros(11)
# s = np.zeros(11)
# p = np.zeros(11)
# h = np.zeros(11)

# # adiabatic efficiency
# e1 = 0.9 # Turbine
# e2 = 0.9 # MC
# e3 = 0.9 # RC

# # Effectiveness of heat exchanger
# epsilon1 = 0.86 # HTR
# epsilon2 = 0.86 # LTR

# fluid = 'CO2'
# p_min = cp.PropsSI('P_CRITICAL', fluid) + 2000
# t_min = cp.PropsSI('T_CRITICAL', fluid) + 5

class syst:
    def __init__(self, fluid, p_max=None, t_max=None, x=None, e1=0.9, e2=0.9, e3=0.9, epsilon1=0.86, epsilon2=0.86):
        self.T = np.zeros(11)
        self.s = np.zeros(11)
        self.p = np.zeros(11)
        self.h = np.zeros(11)
        self. fluid = fluid
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.x = x
        self.p_max = p_max
        self.t_max = t_max
        self.p_min = cp.PropsSI('P_CRITICAL', fluid) + 2000
        self.t_min = cp.PropsSI('T_CRITICAL', fluid) + 1

        self.T[1] = t_max
        self.p[1] = self.p[6] = self.p[7] = self.p[8] = self.p[9] = self.p[10] = p_max
        self.p[5] = self.p[2] = self.p[3] = self.p[4] = self.p_min

        self.h[1] = cp.PropsSI('H', 'P', self.p[1], 'T', self.T[1], self.fluid)
        self.h[2] = ad(p_max, self.h[1], self.p[2], fluid)
        self.h[2] =self.h[1] - self.e1 * (self.h[1] - self.h[2]) # Turbine

        self.h[5] = cp.PropsSI('H', 'P', self.p_min, 'T', self.t_min, self.fluid)
        self.h[6] = ad(self.p[5], self.h[5], self.p[6], self.fluid)
        self.h[6] =self.h[5] +  (self.h[6] - self.h[5])/self.e2 # MC
        self.T[6] = cp.PropsSI('T', 'P', self.p[6], 'H', self.h[6], self.fluid)
        self.T[2] = cp.PropsSI('T', 'P', self.p[2], 'H', self.h[2], self.fluid)


    def hsolve1(self, error = 1e-6):
        # liearization
        a = 0
        b = 0
        c = 0
        iter = 0
        A = np.array([[1,0,0,0,-self.epsilon1],[-1,0,-1,0,1],[1-self.x*self.epsilon2,-1,0,0,0],[1,-1,0,-self.x,0],[0,1-self.x,0,self.x,-1]])
        while True:
            t = np.array([(1-self.epsilon1)*self.h[2]+self.epsilon1*c ,-self.h[2],self.x*self.epsilon2*(a-self.h[6]),-self.x*self.h[6],b*(self.x-1)])
            sol = solve(A, t)
            h4 = sol[1]
            h3 = sol[0]
            h10 = sol[4]
            b1 = (ad(self.p_min, h4, self.p_max, self.fluid)-h4)/self.e3
            c = cp.PropsSI('H', 'P', self.p_min, 'T', cp.PropsSI('T', 'P', self.p_max, 'H', h10, self.fluid), self.fluid) -h10
            a = cp.PropsSI('H', 'P', self.p_max, 'T', cp.PropsSI('T', 'P', self.p_min, 'H', h3, self.fluid), self.fluid) -h3
            if abs(b1-b) < error:
                self.h[3] = sol[0]
                self.h[4] = sol[1]
                self.h[7] = self.h[4] + b1
                self.h[8] = sol[2]
                self.h[9] = sol[3]
                self.h[10] = sol[4]
                break
            b = b1
            iter += 1
            if iter > 800:
                warnings.warn(f'Iteration exceeds 800, x = {self.x}')
                break
        return  (self.h[1]-self.h[2]-self.x*(self.h[6]-self.h[5])-(1-self.x)*(self.h[7]-self.h[4]))/(self.h[1]-self.h[8]), 1-self.x*(self.h[4]-self.h[5])/(self.h[1]-self.h[8])
    
    def hsolve2(self, error = 1e-6):
        # liearization
        b = 0
        c = 0
        iter = 0
        A = np.array([[1,0,0,0,-self.epsilon1],[-1,0,-1,0,1],[self.epsilon2-1,1,0,0,0],[1,-1,0,-self.x,0],[0,1-self.x,0,self.x,-1]])
        while True:
            h6 = cp.PropsSI('H', 'P', self.p_min, 'T', self.T[6], self.fluid)
            t = np.array([(1-self.epsilon1)*self.h[2]+self.epsilon1*c ,-self.h[2],self.epsilon2*h6,-self.x*self.h[6],b*(self.x-1)])
            sol = solve(A, t)
            h4 = sol[1]
            h10 = sol[4]
            b1 = (ad(self.p_min, h4, self.p_max, self.fluid)-h4)/self.e3
            c = cp.PropsSI('H', 'P', self.p_min, 'T', cp.PropsSI('T', 'P', self.p_max, 'H', h10, self.fluid), self.fluid) -h10

            if abs(b1-b) < error:
                self.h[3] = sol[0]
                self.h[4] = sol[1]
                self.h[7] = self.h[4] + b1
                self.h[8] = sol[2]
                self.h[9] = sol[3]
                self.h[10] = sol[4]
                break
            b = b1
            iter += 1
            if iter > 800:
                self.h[3] = sol[0]
                self.h[4] = sol[1]
                self.h[7] = self.h[4] + b1
                self.h[8] = sol[2]
                self.h[9] = sol[3]
                self.h[10] = sol[4]
                warnings.warn(f'Iteration exceeds 800, x = {self.x}')
                break
        return  (self.h[1]-self.h[2]-self.x*(self.h[6]-self.h[5])-(1-self.x)*(self.h[7]-self.h[4]))/(self.h[1]-self.h[8]), 1-self.x*(self.h[4]-self.h[5])/(self.h[1]-self.h[8])
    
    def hsolve3(self, error = 1e-6):
        # liearization
        a = 0
        b = 0
        iter = 0
        A = np.array([[1,0,0,0,-self.epsilon1],[-1,0,-1,0,1],[1-self.x*self.epsilon2,-1,0,0,0],[1,-1,0,-self.x,0],[0,1-self.x,0,self.x,-1]])
        while True:
            t = np.array([self.h[2]-self.epsilon1*cp.PropsSI('H', 'T', self.T[2], 'P', self.p_max, self.fluid) ,-self.h[2],self.x*self.epsilon2*(a-self.h[6]),-self.x*self.h[6],b*(self.x-1)])
            sol = solve(A, t)
            h4 = sol[1]
            h3 = sol[0]
            b1 = (ad(self.p_min, h4, self.p_max, self.fluid)-h4)/self.e3
            a = cp.PropsSI('H', 'P', self.p_max, 'T', cp.PropsSI('T', 'P', self.p_min, 'H', h3, self.fluid), self.fluid) -h3
            if abs(b1-b) < error:
                self.h[3] = sol[0]
                self.h[4] = sol[1]
                self.h[7] = self.h[4] + b1
                self.h[8] = sol[2]
                self.h[9] = sol[3]
                self.h[10] = sol[4]
                break
            b = b1
            iter += 1
            if iter > 800:
                self.h[3] = sol[0]
                self.h[4] = sol[1]
                self.h[7] = self.h[4] + b1
                self.h[8] = sol[2]
                self.h[9] = sol[3]
                self.h[10] = sol[4]
                warnings.warn(f'Iteration exceeds 800, x = {self.x}')
                break
        return  (self.h[1]-self.h[2]-self.x*(self.h[6]-self.h[5])-(1-self.x)*(self.h[7]-self.h[4]))/(self.h[1]-self.h[8]), 1-self.x*(self.h[4]-self.h[5])/(self.h[1]-self.h[8])

#--------------------------------------------------------------------------------------------------------------#

    def hsolve_x2(self, error = 1e-8):
        '''
        This funtion is used to find the value of split ratio which makes the heat capacity of the two inlets in the LTR  equal.
        Coincidentally, the system has maximum efficiency when the split ratio is equal to this value.
        '''
        #initialization
        iter = 0
        t1 = self.x
        self.x = 0.5
        while True:
            self.hsolve1()
            x1 = (self.h[3]-cp.PropsSI('H', 'P', self.p_min, 'T', self.T[6], self.fluid))/(cp.PropsSI('H', 'P', self.p_max, 'T', cp.PropsSI('T', 'P', self.p_min, 'H', self.h[3], self.fluid), self.fluid)-self.h[6])
            if abs(x1-self.x) < error:
                break
            if iter > 800:
                warnings.warn(f'Iteration exceeds 800, x = {self.x}')
                break
            self.x = x1
            iter += 1
        t2 = self.x
        self.x = t1
        return t2

#---------------------------------------------------------------------------------------------------#
    def hsolve_x1(self, error = 1e-8):
        '''
        Find the value of split ratio which makes the heat capacity of the two inlets in the HTR equal. 
        Since the mass flow of the two inlets in the HTR are equal, this happens when their temperature are equal.
        '''
        t1 = self.x
        self.x = 0.5
        min = 0
        max = 1
        iter = 0
        while True:
            self.hsolve1()
            if (self.h[2] - cp.PropsSI('H', 'P', self.p_min, 'T', cp.PropsSI('T', 'H', self.h[10], 'P', self.p_max, self.fluid), self.fluid)) > (cp.PropsSI('H', 'P', self.p_max, 'T', self.T[2], self.fluid) - self.h[10]):
                min = self.x
                self.x = (self.x + max)/2
            else:
                max = self.x
                self.x = (self.x+min)/2
            if abs(max-min) < error:
                break

            if iter > 800:
                warnings.warn(f'Iteration exceeds 800, x = {self.x}')
                break
        t2 = self.x
        self.x = t1
        return t2
#---------------------------------------------------------------------------------------------------#

    def hsolve(self, error = 1e-6):
        x1 = self.hsolve_x1()
        x2 = self.hsolve_x2()

        if self.x < x1:
            eta1, eta2 = self.hsolve3(error = error)
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) < (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
            #     print(f'ERROR1! x = {x}')
            # print('solve3')
        elif self.x < x2:
            eta1, eta2 = self.hsolve1(error = error)
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) > (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
            #     print(f'ERROR1! x = {x}')
            # print('solve1')
        else:
            eta1, eta2 = self.hsolve2(error = error)
            # print('solve2')

        return eta1, eta2

#---------------------------------------------------------------------------------------------------#
def plot_eta(fluid, t_max = 700, p_max = 2e7):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p_max = p_max*1e6
    sys = syst(fluid, t_max = t_max, p_max = p_max)
    x1 = sys.hsolve_x1()
    x2 = sys.hsolve_x2()
    x = np.linspace(max(x1,0.3), 1, 100)

    print(f"x1 = {x1}, x2 = {x2}")
    eta = np.array([]) 
    for i in x:
        sys.x = i
        if i < x1:
            eta = np.append(eta, sys.hsolve3()[0])
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) - (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]) < 1e-3:
            #     print(f'ERROR1! x = {i}')
            # if (h[3] - cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)) < (i*(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])):
            #     print(f'ERROR2! x = {i}')
        elif i < x2:
            eta = np.append(eta, sys.hsolve1()[0])
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) - (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10])> 1e-3:
            #     print(f'ERROR1! x = {i}')
            # if (h[3] - cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)) < (i*(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])):
            #     print(f'ERROR2! x = {i}')
        else:
            eta = np.append(eta, sys.hsolve2()[0])
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) > (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
            #     print(f'ERROR1! x = {i}')
            # if (h[3] - cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)) > (i*(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])):
            #     print(f'ERROR2! x = {i}')
    ax.plot(x, eta)
    ax.set_xlabel('Split Ratio')
    ax.set_ylabel('Efficiency')
    ax.legend(title = f'$\mathrm{{p}}_{{\mathrm{{max}}}}$ : {p_max/1e6} MPa \n$\mathrm{{T}}_{{\mathrm{{max}}}}$ : {t_max} K')

    return fig


def find_max(fluid, t_max=900, p1=15, p2=30):
    eta = np.array([])
    max_eta = 0
    index = 0
    s = 0
    p = np.linspace(p1,p2,100)
    for i in p:
        sys = syst(fluid=fluid, t_max=t_max, p_max=i*1e6)
        x = sys.hsolve_x2()
        eta = np.append(eta, 1-x*(sys.h[4]-sys.h[5])/(sys.h[1]-sys.h[8]))
        if eta[-1] > max_eta:
            max_eta = eta[-1]
            index = i
            s = x
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(p, eta)
    ax.set_xlabel('Maximum Pressure (MPa)')
    ax.set_ylabel('Efficiency')
    ax.set_title('Maximum Efficiency')
    ax.legend(title = f'Maximum Temperature: {t_max:.2f} K\nMaximum Efficiency: {max_eta:.4f}\nOptimal Pressure: {index:.2f} MPa\nsplit ratio: {s:.2f}')
    return fig


# def plot_para():
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for j in np.linspace(1,3,5):
#         x = np.linspace(0.4, 0.78, 250)
#         eta = np.array([]) 
#         for i in x:
#             eta = np.append(eta, hsolve(i, j*1e7,700,fluid)[0])
#         ax.plot(x, eta, label = f'p_max = {j*10} MPa')
#     ax.set_xlabel('Split Ratio')
#     ax.set_ylabel('Efficiency')



def plot_system(x, p_max, t_max, fluid):
    p_max = p_max*1e6
    fig1 = plt.figure(dpi=150)
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(dpi=150)
    ax2 = fig2.add_subplot(111)

    sys = syst(fluid=fluid, t_max=t_max, p_max=p_max, x=x)
    eta1, eta2 = sys.hsolve()


    for i in range(1,11):
        sys.s[i] = cp.PropsSI('S', 'P', sys.p[i], 'H', sys.h[i], fluid)
        sys.T[i] = cp.PropsSI('T', 'P', sys.p[i], 'H', sys.h[i], fluid)

    if sys.T[2] < sys.T[3]:
        warnings.warn('The temperature of the hot stream is lower than that of the cold stream, reset the split ratio.', UserWarning)

    for i in range(1,11):
        print(f'T{i} = {sys.T[i]} K', f's{i} = {sys.s[i]} J/kgK, h{i} = {sys.h[i]} J/kg, p{i} = {sys.p[i]} Pa')
        ax1.plot(sys.s[i],sys.T[i],'o',color = 'red')
        ax1.annotate(i, xy=(sys.s[i], sys.T[i]), 
                     xytext=(sys.s[i] -30, sys.T[i]+10),
                     fontsize=6,
                     bbox=dict(boxstyle='circle', fc='white', ec='black'))

    ax1.set_xlabel('Entropy (J/kgK)')
    ax1.set_ylabel('Temperature (K)')

    # Plot the table of the system
    df = pd.DataFrame({'Temperature [K]':np.round(sys.T[1:],2), 'Entropy [J/(Kg K)]':np.round(sys.s[1:],2), 'Enthalpy [J/Kg]':np.round(sys.h[1:],2), 'Pressure [MPa]':np.round(sys.p[1:]/1e6,2)})
    group_number = np.arange(1,11)
    df.insert(loc=0, column='State', value=np.round(group_number,0))

    ax2.axis('off')
    ax2.table(cellText=df.values, colLabels=df.columns, loc='center')

    # Plot the temperature-entropy curve of carbon dioxide under saturated state
    plot_TS_diagram(fluid, 216.492, sys.t_min,ax1)
    plot_isobar_TS_diagram(fluid,sys.p[2],sys.T[5],sys.T[2],ax1, label='isobaric process')
    plot_isobar_TS_diagram(fluid,sys.p[1],sys.T[6],sys.T[1],ax1)
    plot_isothermal_TS_diagram(sys.s[2], sys.s[1],sys.T[2],sys.T[1],ax1,label='isothermal process')
    plot_isothermal_TS_diagram(sys.s[5],sys.s[6],sys.T[5],sys.T[6],ax1)
    plot_isothermal_TS_diagram(sys.s[4], sys.s[7],sys.T[4],sys.T[7],ax1)
    ax1.set_title(f'T-s diagram of the Brayton cycle')
    ax1.set_xlabel('Entropy (J/kgK)')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend(title=f'$\eta$ = {eta1:.5f}\n$p_{{max}}$ = {p_max/1e6:.2f} MPa\n$p_{{min}}$ = {sys.p_min/1e6:.2f} MPa\n$T_{{max}}$ = {t_max:.2f} K\n$T_{{min}}$ = {sys.t_min:.2f} K\n$x$ = {x:.2f}\n',
               loc='best')
    
    print(f'eta1 = {eta1:.8f}, eta2 = {eta2:.8f}')
    return fig1, fig2


def heat_exchanger(fluid, x, p_max, t_max, Q, k1=4000, k2=4000, plot=False):
    p_max = p_max*1e6
    Q = Q*1e3
    sys = syst(fluid=fluid, t_max=t_max, p_max=p_max, x=x)
    sys.hsolve()
    v = sys.h[4] - x*sys.h[6]
    u = sys.h[2] - sys.h[10]
    #calculate the mass flow of the system
    m = Q/(sys.h[1] - sys.h[8])

    def t1(h):
        return cp.PropsSI('T', 'H', h, 'P', sys.p_min, 'CO2')

    def t2(h):
        return cp.PropsSI('T', 'H', h, 'P', p_max, 'CO2')

    def f(h):
        return 1/(t2((h-v)/x) - t1(h))
    
    def g(h):
        return 1/(t2((h-v)) - t1(h))
    fig1, ax1 = plt.subplots(dpi=150)
    fig2, ax2 = plt.subplots(dpi=150)
    if not plot:
        return -quad(f, sys.h[4], sys.h[3])[0]/k1*m, -quad(g, sys.h[3], sys.h[2])[0]/k2*m, fig1, fig2
    else:
        S = np.array([])
        m1 = np.array([])
        m2 = np.array([])
        for i in np.linspace(sys.h[4], sys.h[3], 100):
            s = -quad(f, i, sys.h[3])[0]/k1*m
            S = np.append(S, s)
            m1 = np.append(m1, t1(i))
            m2 = np.append(m2, t2((i-v)/x))

        ax1.plot(S, m2)
        ax1.plot(S, m1)
        ax1.set_xlabel('A ($\mathrm{m}^2$)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('LTR')
        ax1.legend(title='k = {:.2f} J/($\mathrm{{m}}^{{2}}$K)\nA = {:.2f} $\mathrm{{m}}^2$'.format(k1, S[0]))
        ax1.annotate('({:.2f}, {:.2f})'.format(S[-1], m2[-1]), xy=(S[-1], m2[-1]), xytext = (-40, -10), textcoords='offset points', fontsize=8)
        ax1.annotate('({:.2f}, {:.2f})'.format(S[0], m1[0]), xy=(S[0], m1[0]), xytext = (-40, -10), textcoords='offset points', fontsize=8)

        T = np.array([])
        l1 = np.array([])
        l2 = np.array([])
        for i in np.linspace(sys.h[3],sys.h[2], 100):
            t = -quad(g, i, sys.h[2])[0]/k2*m
            T = np.append(T, t)
            l1 = np.append(l1, t1(i))
            l2 = np.append(l2, t2((i-u)))

        ax2.plot(T, l2)
        ax2.plot(T, l1)
        ax2.set_xlabel('A ($\mathrm{m}^2$)')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_title('HTR')
        ax2.legend(title='k = {:.2f} J/($\mathrm{{m}}^{{2}}$K)\nA = {:.2f} $\mathrm{{m}}^2$'.format(k2, T[0]))
        ax2.annotate('({:.2f}, {:.2f})'.format(T[-1], l2[-1]), xy=(T[-1], l2[-1]), xytext = (-40, -10), textcoords='offset points', fontsize=8)
        ax2.annotate('({:.2f}, {:.2f})'.format(T[0], l1[0]), xy=(T[0], l1[0]), xytext = (-40, -10), textcoords='offset points', fontsize=8)
        return S[0], T[0], fig1, fig2
    
