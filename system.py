import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
from scipy.linalg import solve
import pandas as pd
from scipy.integrate import quad
import matplotlib
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

T = np.zeros(11)
s = np.zeros(11)
p = np.zeros(11)
h = np.zeros(11)

# adiabatic efficiency
e1 = 0.9 # Turbine
e2 = 0.9 # MC
e3 = 0.9 # RC

# Effectiveness of heat exchanger
epsilon1 = 0.86 # HTR
epsilon2 = 0.86 # LTR

fluid = 'CO2'
p_min = cp.PropsSI('P_CRITICAL', fluid) + 2000
t_min = cp.PropsSI('T_CRITICAL', fluid) + 1

def get_props(fluid):
    global p_min, t_min
    p_min = cp.PropsSI('P_CRITICAL', fluid) + 2000
    t_min = cp.PropsSI('T_CRITICAL', fluid) + 1

#---------------------------------------------------------------------------------------------------#
def hsolve_x2(p_max, t_max, fluid, error = 1e-8):
    '''
    This funtion is used to find the value of split ratio which makes the heat capacity of the two inlets in the LTR  equal.
    Coincidentally, the system has maximum efficiency when the split ratio is equal to this value.
    '''
    #initialization
    x = 0.5
    while True:
        hsolve1(x, p_max, t_max, fluid)
        x1 = (h[3]-cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid))/(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])
        if abs(x1-x) < error:
            break
        x = x1
    return x

#---------------------------------------------------------------------------------------------------#
def hsolve_x1(p_max, t_max, fluid, error = 1e-8):
    '''
    Find the value of split ratio which makes the heat capacity of the two inlets in the HTR equal. 
    Since the mass flow of the two inlets in the HTR are equal, this happens when their temperature are equal.
    '''
    x = 0.5
    min = 0
    max = 1
    while True:
        hsolve1(x, p_max, t_max, fluid, error)
        if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) > (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
            min = x
            x = (x + max)/2
        else:
            max = x
            x = (x+min)/2
        if abs(max-min) < error:
            break
    # print(max - min)
    return x
#---------------------------------------------------------------------------------------------------#

def hsolve1(x, p_max, t_max, fluid, error = 1e-6):
    # Initializtion
    T[1] = t_max
    p[1] = p[6] = p[7] = p[8] = p[9] = p[10] = p_max
    p[5] = p[2] = p[3] = p[4] = p_min

    h[1] = cp.PropsSI('H', 'P', p[1], 'T', T[1], fluid)
    h[2] = ad(p_max, h[1], p[2], fluid)
    h[2] = h[1] - e1 * (h[1] - h[2]) # Turbine

    h[5] = cp.PropsSI('H', 'P', p_min, 'T', t_min, fluid)
    h[6] = ad(p[5], h[5], p[6], fluid)
    h[6] = h[5] +  (h[6] - h[5])/e2 # MC
    T[6] = cp.PropsSI('T', 'P', p[6], 'H', h[6], fluid)
    T[2] = cp.PropsSI('T', 'P', p[2], 'H', h[2], fluid)

    # liearization
    a = 0
    b = 0
    c = 0
    iter = 0
    A = np.array([[1,0,0,0,-epsilon1],[-1,0,-1,0,1],[1-x*epsilon2,-1,0,0,0],[1,-1,0,-x,0],[0,1-x,0,x,-1]])
    while True:
        t = np.array([(1-epsilon1)*h[2]+epsilon1*c ,-h[2],x*epsilon2*(a-h[6]),-x*h[6],b*(x-1)])
        sol = solve(A, t)
        h4 = sol[1]
        h3 = sol[0]
        h10 = sol[4]
        b1 = (ad(p_min, h4, p_max, fluid)-h4)/e3
        c = cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'P', p_max, 'H', h10, fluid), fluid) -h10
        a = cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h3, fluid), fluid) -h3
        if abs(b1-b) < error:
            h[3] = sol[0]
            h[4] = sol[1]
            h[7] = h[4] + b1
            h[8] = sol[2]
            h[9] = sol[3]
            h[10] = sol[4]
            break
        b = b1
        iter += 1
        if iter > 800:
            print(f'Error in hsolve1: iteration exceeds 800, x = {x}')
            break
    return  (h[1]-h[2]-x*(h[6]-h[5])-(1-x)*(h[7]-h[4]))/(h[1]-h[8]), 1-x*(h[4]-h[5])/(h[1]-h[8])

def hsolve2(x, p_max, t_max, fluid, error = 1e-6):
    # Initializtion
    T[1] = t_max
    p[1] = p[6] = p[7] = p[8] = p[9] = p[10] = p_max
    p[5] = p[2] = p[3] = p[4] = p_min

    h[1] = cp.PropsSI('H', 'P', p[1], 'T', T[1], fluid)
    h[2] = ad(p_max, h[1], p[2], fluid)
    h[2] = h[1] - e1 * (h[1] - h[2]) # Turbine

    h[5] = cp.PropsSI('H', 'P', p_min, 'T', t_min, fluid)
    h[6] = ad(p[5], h[5], p[6], fluid)
    h[6] = h[5] +  (h[6] - h[5])/e2 # MC
    T[6] = cp.PropsSI('T', 'P', p[6], 'H', h[6], fluid)
    T[2] = cp.PropsSI('T', 'P', p[2], 'H', h[2], fluid)

    # liearization
    b = 0
    c = 0
    iter = 0
    A = np.array([[1,0,0,0,-epsilon1],[-1,0,-1,0,1],[epsilon2-1,1,0,0,0],[1,-1,0,-x,0],[0,1-x,0,x,-1]])
    while True:
        h6 = cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)
        t = np.array([(1-epsilon1)*h[2]+epsilon1*c ,-h[2],epsilon2*h6,-x*h[6],b*(x-1)])
        sol = solve(A, t)
        h4 = sol[1]
        h10 = sol[4]
        b1 = (ad(p_min, h4, p_max, fluid)-h4)/e3
        c = cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'P', p_max, 'H', h10, fluid), fluid) -h10

        if abs(b1-b) < error:
            h[3] = sol[0]
            h[4] = sol[1]
            h[7] = h[4] + b1
            h[8] = sol[2]
            h[9] = sol[3]
            h[10] = sol[4]
            break
        b = b1
        iter += 1
        if iter > 800:
            print(f'Error hsolve2: iteration exceeds 800, x = {x}')
            break
    return  (h[1]-h[2]-x*(h[6]-h[5])-(1-x)*(h[7]-h[4]))/(h[1]-h[8]), 1-x*(h[4]-h[5])/(h[1]-h[8])

def hsolve3(x, p_max, t_max, fluid, error = 1e-6):
    # Initializtion
    T[1] = t_max
    p[1] = p[6] = p[7] = p[8] = p[9] = p[10] = p_max
    p[5] = p[2] = p[3] = p[4] = p_min

    h[1] = cp.PropsSI('H', 'P', p[1], 'T', T[1], fluid)
    h[2] = ad(p_max, h[1], p[2], fluid)
    h[2] = h[1] - e1 * (h[1] - h[2]) # Turbine

    h[5] = cp.PropsSI('H', 'P', p_min, 'T', t_min, fluid)
    h[6] = ad(p[5], h[5], p[6], fluid)
    h[6] = h[5] +  (h[6] - h[5])/e2 # MC
    T[6] = cp.PropsSI('T', 'P', p[6], 'H', h[6], fluid)
    T[2] = cp.PropsSI('T', 'P', p[2], 'H', h[2], fluid)

    # liearization
    a = 0
    b = 0
    iter = 0
    A = np.array([[1,0,0,0,-epsilon1],[-1,0,-1,0,1],[1-x*epsilon2,-1,0,0,0],[1,-1,0,-x,0],[0,1-x,0,x,-1]])
    while True:
        t = np.array([h[2]-epsilon1*cp.PropsSI('H', 'T', T[2], 'P', p_max, fluid) ,-h[2],x*epsilon2*(a-h[6]),-x*h[6],b*(x-1)])
        sol = solve(A, t)
        h4 = sol[1]
        h3 = sol[0]
        b1 = (ad(p_min, h4, p_max, fluid)-h4)/e3
        a = cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h3, fluid), fluid) -h3
        if abs(b1-b) < error:
            h[3] = sol[0]
            h[4] = sol[1]
            h[7] = h[4] + b1
            h[8] = sol[2]
            h[9] = sol[3]
            h[10] = sol[4]
            break
        b = b1
        iter += 1
        if iter > 800:
            print(f'Error hsolve3: iteration exceeds 800, x = {x}')
            break
    return  (h[1]-h[2]-x*(h[6]-h[5])-(1-x)*(h[7]-h[4]))/(h[1]-h[8]), 1-x*(h[4]-h[5])/(h[1]-h[8])


def hsolve(x, p_max, t_max, fluid, error = 1e-6):
    get_props(fluid=fluid)
    x1 = hsolve_x1(p_max, t_max, fluid)
    x2 = hsolve_x2(p_max, t_max, fluid)

    if x < x1:
        eta1, eta2 = hsolve3(x, p_max, t_max, fluid, error = error)
        # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) < (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
        #     print(f'ERROR1! x = {x}')
        # print('solve3')
    elif x < x2:
        eta1, eta2 = hsolve1(x, p_max, t_max, fluid, error = error)
        # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) > (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
        #     print(f'ERROR1! x = {x}')
        # print('solve1')
    else:
        eta1, eta2 = hsolve2(x, p_max, t_max, fluid, error = error)
        # print('solve2')

    return eta1, eta2

#---------------------------------------------------------------------------------------------------#
def plot_eta(fluid, t_max = 700, p_max = 2e7):
    get_props(fluid=fluid)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x1 = hsolve_x1(p_max, t_max, fluid)
    x2 = hsolve_x2(p_max, t_max, fluid)
    if x1 > 0.3:
        x = np.linspace(x1, 1, 200)
    else:
        x = np.linspace(0.3, 1, 200)
    print(f"x1 = {x1}, x2 = {x2}")
    eta = np.array([]) 
    for i in x:
        if i < x1:
            eta = np.append(eta, hsolve3(i, p_max, t_max, fluid)[0])
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) - (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]) < 1e-3:
            #     print(f'ERROR1! x = {i}')
            # if (h[3] - cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)) < (i*(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])):
            #     print(f'ERROR2! x = {i}')
        elif i < x2:
            eta = np.append(eta, hsolve1(i, p_max, t_max, fluid)[0])
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) - (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10])> 1e-3:
            #     print(f'ERROR1! x = {i}')
            # if (h[3] - cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)) < (i*(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])):
            #     print(f'ERROR2! x = {i}')
        else:
            eta = np.append(eta, hsolve2(i, p_max, t_max, fluid)[0])
            # if (h[2] - cp.PropsSI('H', 'P', p_min, 'T', cp.PropsSI('T', 'H', h[10], 'P', p_max, fluid), fluid)) > (cp.PropsSI('H', 'P', p_max, 'T', T[2], fluid) - h[10]):
            #     print(f'ERROR1! x = {i}')
            # if (h[3] - cp.PropsSI('H', 'P', p_min, 'T', T[6], fluid)) > (i*(cp.PropsSI('H', 'P', p_max, 'T', cp.PropsSI('T', 'P', p_min, 'H', h[3], fluid), fluid)-h[6])):
            #     print(f'ERROR2! x = {i}')
    ax.plot(x, eta, label = f'$\mathrm{{p}}_{{\mathrm{{max}}}}$ : {p_max/1e6} MPa \n$\mathrm{{T}}_{{\mathrm{{max}}}}$ : {t_max} K')
    ax.set_xlabel('Split Ratio')
    ax.set_ylabel('Efficiency')
    ax.legend()
    return fig

def plot_p(fluid, x = 0.65, t_max = 700):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    eta = np.array([])
    p_max = np.linspace(1, 3, 100)
    for i in p_max:
        eta = np.append(eta, hsolve(x, i*1e7, t_max, fluid)[0])
    ax.plot(p_max, eta, label = f'Split Ratio: {x} \nMaximum Temperature: {t_max} K')
    ax.set_xlabel('Maximum Pressure (10MPa)')
    ax.set_ylabel('Efficiency')

def find_max(fluid, t_max=900, p1=15, p2=30):
    get_props(fluid=fluid)
    eta = np.array([])
    max_eta = 0
    index = 0
    s = 0
    p = np.linspace(p1,p2,100)
    for i in p:
        x = hsolve_x2(i*1e6, t_max, fluid)
        eta = np.append(eta, hsolve2(x, i*1e6, t_max, fluid)[0])
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
    # print(f'Maximum Efficiency: {np.max(eta)}')
    # print(f'Optimal Pressure: {np.linspace(1.5,2.5,100)[np.argmax(eta)]*1e7} Pa')
    # print(f'Optimal Split Ratio: {hsolve_x2(np.linspace(1.5,2.5,100)[np.argmax(eta)]*1e7, t_max, fluid)}')
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

    eta1, eta2 = hsolve(x, p_max, t_max, fluid)

    for i in range(1,11):
        s[i] = cp.PropsSI('S', 'P', p[i], 'H', h[i], fluid)
        T[i] = cp.PropsSI('T', 'P', p[i], 'H', h[i], fluid)

    for i in range(1,11):
        print(f'T{i} = {T[i]} K', f's{i} = {s[i]} J/kgK, h{i} = {h[i]} J/kg, p{i} = {p[i]} Pa')
        ax1.plot(s[i],T[i],'o',color = 'red')
        ax1.annotate(i, xy=(s[i], T[i]), 
                     xytext=(s[i] -30, T[i]+10),
                     fontsize=6,
                     bbox=dict(boxstyle='circle', fc='white', ec='black'))

    ax1.set_xlabel('Entropy (J/kgK)')
    ax1.set_ylabel('Temperature (K)')

    # Plot the table of the system
    df = pd.DataFrame({'Temperature [K]':np.round(T[1:],2), 'Entropy [J/(Kg K)]':np.round(s[1:],2), 'Enthalpy [J/Kg]':np.round(h[1:],2), 'Pressure [MPa]':np.round(p[1:]/1e6,2)})
    group_number = np.arange(1,11)
    df.insert(loc=0, column='State', value=np.round(group_number,0))

    ax2.axis('off')
    ax2.table(cellText=df.values, colLabels=df.columns, loc='center')

    # Plot the temperature-entropy curve of carbon dioxide under saturated state
    plot_TS_diagram(fluid, 216.492, t_min,ax1)
    plot_isobar_TS_diagram(fluid,p[2],T[5],T[2],ax1, label='isobaric process')
    plot_isobar_TS_diagram(fluid,p[1],T[6],T[1],ax1)
    plot_isothermal_TS_diagram(s[2], s[1],T[2],T[1],ax1,label='isothermal process')
    plot_isothermal_TS_diagram(s[5],s[6],T[5],T[6],ax1)
    plot_isothermal_TS_diagram(s[4], s[7],T[4],T[7],ax1)
    ax1.set_title(f'T-s diagram of the Brayton cycle')
    ax1.set_xlabel('Entropy (J/kgK)')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend(title=f'$\eta$ = {eta1:.2f}\n$p_{{max}}$ = {p_max/1e6:.2f} MPa\n$p_{{min}}$ = {p_min/1e6:.2f} MPa\n$T_{{max}}$ = {t_max:.2f} K\n$T_{{min}}$ = {t_min:.2f} K\n$x$ = {x:.2f}\n',
               loc='best')
    
    print(f'eta1 = {eta1:.8f}, eta2 = {eta2:.8f}')
    return fig1, fig2

def heat_exchanger(fluid, x, p_max, t_max, Q, k1=4000, k2=4000, plot=False):
    p_max = p_max*1e6
    Q = Q*1e3
    hsolve(x, p_max, t_max, fluid)
    v = h[4] - x*h[6]
    u = h[2] - h[10]
    #calculate the mass flow of the system
    m = Q/(h[1] - h[8])

    def t1(h):
        return cp.PropsSI('T', 'H', h, 'P', p_min, 'CO2')

    def t2(h):
        return cp.PropsSI('T', 'H', h, 'P', p_max, 'CO2')

    def f(h):
        return 1/(t2((h-v)/x) - t1(h))
    
    def g(h):
        return 1/(t2((h-v)) - t1(h))
    fig1, ax1 = plt.subplots(dpi=150)
    fig2, ax2 = plt.subplots(dpi=150)
    if not plot:
        return -quad(f, h[4], h[3])[0]/k1*m, -quad(g, h[3], h[2])[0]/k2*m, fig1, fig2
    else:
        S = np.array([])
        m1 = np.array([])
        m2 = np.array([])
        for i in np.linspace(h[4], h[3], 100):
            s = -quad(f, i, h[3])[0]/k1*m
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
        for i in np.linspace(h[3], h[2], 100):
            t = -quad(g, i, h[2])[0]/k2*m
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
    
# if __name__ == '__main__':
    # plot_system(0.74, 2.788e7, 900, fluid, Q=277e3, plot_heat_exchanger=True)
    # plot_eta(fluid, t_max = 900, p_max=2.788e7)
    # find_max(fluid, t_max=900)



