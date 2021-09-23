#!/usr/bin/env python
# coding: utf-8

# In[3]:


import rebound
import reboundx
import numpy as np

### DEFINITIONS

radeg = np.pi/180

def add_L4(sim, number):
    a_rand = np.random.normal(38,2,size=number)
    a_rand = a_rand/100 + 5
    e_rand = np.random.normal(9,2,size=number)
    e_rand = e_rand/100
    w_rand = np.random.normal(0,4,size=number)*radeg
    half = int(number/2)
    i_rand1 = np.random.normal(9,4,size=half+1)*radeg
    i_rand2 = np.random.normal(-9,4,size=half)*radeg
    i_rand = np.concatenate((i_rand1,i_rand2))
    f_val = 60*radeg
        
    for i in range(number):
        sem = a_rand[i]
        ecc = e_rand[i]
        icl = i_rand[i]
        Ome = w_rand[i]
        has = 'L4 {0}'.format(i)
        sim.add(m=0, primary=sim.particles['Sun'], a=sem, e=ecc, inc=icl, Omega=Ome, f=f_val, hash=has)
    return

def add_L5(sim, number):
    a_rand = np.random.normal(38,2,size=number)
    a_rand = a_rand/100 + 5
    e_rand = np.random.normal(9,2,size=number)
    e_rand = e_rand/100
    w_rand = np.random.normal(0,4,size=number)*radeg
    half = int(number/2)
    i_rand1 = np.random.normal(9,4,size=half+1)*radeg
    i_rand2 = np.random.normal(-9,4,size=half)*radeg
    i_rand = np.concatenate((i_rand1,i_rand2))
    f_val = -60*radeg

    
    for i in range(number):
        sem = a_rand[i]
        ecc = e_rand[i]
        icl = i_rand[i]
        Ome = w_rand[i]
        has = 'L5 {0}'.format(i)
        sim.add(m=0, primary=sim.particles['Sun'], a=sem, e=ecc, inc=icl, Omega=Ome, f=f_val, hash=has)
    return

def masses(x):
    # for input array of time values, approximate M_star (in M_sol) at those times in its life
    y = np.zeros_like(x)
    for i, time in enumerate(x):
        if (time <= 1.132e10):
            y[i] = 1
        elif (1.132e10 < time <= 1.1336e10):
            y[i] = 0.05 * (708.5 - time/(1.6e7))**(1/3) + .95
        elif (1.1336e10 < time <= 1.1463e10):
            y[i] =  -8**((time - 1.1463e10)/574511)/2.4 + .95
        elif (1.1463e10 < time):
            y[i] = 0.532
    return y

def lums_array(x):
    # for input array of time values, approximate log(L_star) (in log(L_sol)) at those times
    y = np.zeros_like(x)
    for i, time in enumerate(x):
        if (time <= 1.113e10):
            y[i] = 1.05
        elif (1.113e10 < time <= 1.1225e10):
            y[i] = 1.45 + ((1.45 - 1.1)/(1.1225e10 - 1.1135e10))*(time - 1.1225e10)
        elif (1.1225e10 < time <= 1.125e10):
            y[i] = 1.45
        elif (1.125 < time <= 1.1336e10):
            y[i] = 1.35 + .1*1.002**((time - 1.125e10)/58000)
        elif (1.1336e10 < time <= 1.142e10):
            y[i] = 1.673
        elif (1.142e10 < time <= 1.14397e10):
            y[i] = 3.198e-9*time - 34.85
        elif (1.14397e10 < time <= 1.14479e10):
            y[i] = 1.736 + 0.032*1.5**((time - 1.14455e10)/360000)
        elif (1.14479e10 < time <= 1.1462e10):
            y[i] = 2.15 + 0.00021*1.5**((time - 1.1444e10)/870000)
        elif (1.1462e10 < time <= 1.14632e10):
            y[i] = 3.5 + (.43/0.0001e10)*(time - 1.1463e10)
        elif (1.14632e10 < time <= 1.14636e10):
            y[i] = 2.3*((time - 1.1463e10)/45000)**(-0.3)
        elif (1.14636e10 < time <= 1.14654715e10):
            y[i] = .2 + ((.2 - 1.05)/(1.14654715e10 - 1.14636e10))*(time - 1.14654715e10)
        elif (1.14654715e10 < time):
            y[i] = .2
    return y        

def inst_lum(x):
    # for a single time input, output log(L_star) (in log(L_sol)) at that time
    time = x
    if (time <= 1.113e10):
        y = 1.05
    elif (1.113e10 < time <= 1.1225e10):
        y = 1.45 + ((1.45 - 1.1)/(1.1225e10 - 1.1135e10))*(time - 1.1225e10)
    elif (1.1225e10 < time <= 1.125e10):
        y = 1.45
    elif (1.125 < time <= 1.1336e10):
        y = 1.35 + .1*1.002**((time - 1.125e10)/58000)
    elif (1.1336e10 < time <= 1.142e10):
        y = 1.673
    elif (1.142e10 < time <= 1.14397e10):
        y = 3.198e-9*time - 34.85
    elif (1.14397e10 < time <= 1.14479e10):
        y = 1.736 + 0.032*1.5**((time - 1.14455e10)/360000)
    elif (1.14479e10 < time <= 1.1462e10):
        y = 2.15 + 0.00021*1.5**((time - 1.1444e10)/870000)
    elif (1.1462e10 < time <= 1.14632e10):
        y = 3.5 + (.43/0.0001e10)*(time - 1.1463e10)
    elif (1.14632e10 < time <= 1.14636e10):
        y = 2.3*((time - 1.1463e10)/45000)**(-0.3)
    elif (1.14636e10 < time <= 1.14654715e10):
        y = .2 + ((.2 - 1.05)/(1.14654715e10 - 1.14636e10))*(time - 1.14654715e10)
    elif (1.14654715e10 < time):
        y = .2
    return y

def yark(simp, rebx_force, particles, N):
    sim = simp.contents
    part = sim.particles
    
    current_time = sim.t + T0
    L_sol = np.exp(inst_lum(current_time))*0.000235 # solar luminosity in au^2 M_sol/yr^3
    
    sim.move_to_hel()
    for troj in range(num_tr):
    
        i = troj + 1
        
        x = part[i].x  ; y = part[i].y  ; z = part[i].z
        R = troj_radii[i-1]
        m_ast = troj_masses[i-1]
    
        c = 63197.8 # speed of light in au/yr
        r = np.sqrt(x**2 + y**2 + z**2)
        A = (R**2 * L_sol)/(4*m_ast*c)
    
        part[i].ax += (A/r**3) * (x + 0.25*y)
        part[i].ay += (A/r**3) * (y + 0.25*x)
        part[i].az += (A/r**3) * z
    return

def roch2(x,y,M1,M2,a):
    q = M2/M1
    x /= a
    y /= a
    s1 = np.sqrt(x**2 + y**2)
    s2 = np.sqrt((x-1)**2 + y**2)
    
    term1 = 2/(s1*(1+q))
    term2 = 2*q/(s2*(1+q))
    term3 = (x - q/(1+q))**2
    term4 = y**2
    return term1 + term2 + term3 + term4

def cart_to_pol(x,y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return r, phi

def pol_to_cart(r,phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y

def L45(msun,mjup):
    u2 = mjup/(msun+mjup)
    
    x_L4 = 0.5 - u2
    x_L5 = x_L4
    
    y_L4 = np.sqrt(3)/2
    y_L5 = -y_L4
    
    return np.array([x_L4,x_L5]), np.array([y_L4,y_L5])

###########################
###########################
###########################

### SIMULATION

###########################
###########################
###########################

'''
Outputs:
  -- Mass details
  -- Four numpy files containing details of evolution of orbits
''' 

num_L4 = 125
num_L5 = num_L4
T0 = 1.14610e10
t_tot = 2500000

N_times = 10000
ts = np.linspace(0, t_tot, N_times)
mtimes = masses(ts + T0)
lumins = lums_array(ts + T0)

sim = rebound.Simulation()

radeg = np.pi/180

M0 = mtimes[0]
num_tr = num_L4 + num_L5

sim.add(m=M0,x=0, y=0, z=0, vx=0, vy=0, vz=0, hash='Sun')
add_L4(sim, num_L4)
add_L5(sim, num_L5)
sim.add(m=9.543e-4, a=5.2, e=.04839, inc=.022689, Omega=0, omega=0, hash='jupiter')

sim.integrator = 'whfast'
sim.dt = 0.5
sim.move_to_com()

ps = sim.particles

Nout = 100000
times = np.linspace(0,t_tot,Nout)
mstar = np.zeros(Nout)

rebx = reboundx.Extras(sim)

starmass = reboundx.Interpolator(rebx, ts, mtimes, 'spline')

yrkv = rebx.create_force("yarkovsky")

yrkv.force_type = "pos"
yrkv.update_accelerations = yark
rebx.add_force(yrkv)

rad_ast = 1                                         # radius in km
troj_radii = np.full(num_tr, rad_ast/1.496e+8)      # gives each asteroid a radius in AU

mass_typic = 3*(4/3)*np.pi*(rad_ast*100000)**3      # gives typical mass @ this radius, w/ density = 3 g cm^-3
troj_masses = np.random.normal(mass_typic, .3*mass_typic, num_tr)        
                                                    # gives array of values around that mass
troj_masses /= 1.9891e33                            # divides each mass by M_sol to get masses in M_sol

mass = np.zeros(Nout)

x_sol = np.zeros(Nout); y_sol = np.zeros(Nout); z_sol = np.zeros(Nout)
x_sol[0] = ps['Sun'].x
y_sol[0] = ps['Sun'].y
z_sol[0] = ps['Sun'].z

x_jup = np.zeros(Nout); y_jup = np.zeros(Nout); z_jup = np.zeros(Nout)
x_jup[0] = ps['jupiter'].x
y_jup[0] = ps['jupiter'].y
z_jup[0] = ps['jupiter'].z

a_jup = np.zeros(Nout)
e_jup = np.zeros(Nout) 
i_jup = np.zeros(Nout)
pmjup = np.zeros(Nout)
lmjup = np.zeros(Nout)

a_jup[0] = ps['jupiter'].a
e_jup[0] = ps['jupiter'].e
i_jup[0] = ps['jupiter'].inc
pmjup[0] = ps['jupiter'].pomega
lmjup[0] = ps['jupiter'].l

a_vals = np.zeros((num_tr, Nout))
e_vals = np.zeros((num_tr, Nout))
i_vals = np.zeros((num_tr, Nout))
pmvals = np.zeros((num_tr, Nout))
lmvals = np.zeros((num_tr, Nout))

x_vals = np.zeros((num_tr, Nout))
y_vals = np.zeros((num_tr, Nout))

for moon in range(num_L4):
    a_vals[moon,0] = ps['L4 {0}'.format(moon)].a
    e_vals[moon,0] = ps['L4 {0}'.format(moon)].e
    i_vals[moon,0] = ps['L4 {0}'.format(moon)].inc
    lmvals[moon,0] = ps['L4 {0}'.format(moon)].l
    pmvals[moon,0] = ps['L4 {0}'.format(moon)].pomega
    x_vals[moon,0] = ps['L4 {0}'.format(moon)].x
    y_vals[moon,0] = ps['L4 {0}'.format(moon)].y

for moon in range(num_L5):
    a_vals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].a
    e_vals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].e
    i_vals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].inc
    lmvals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].l
    pmvals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].pomega
    x_vals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].x
    y_vals[moon + num_L4,0] = ps['L5 {0}'.format(moon)].y

###########################
###########################
###########################

### RUNNING

###########################
###########################
###########################

for i, time in enumerate(times):
    sim.integrate(time)

    ps[0].m = starmass.interpolate(rebx, t=sim.t)
    sim.move_to_com()

    mass[i] = ps['Sun'].m
    x_sol[i] = ps['Sun'].x
    y_sol[i] = ps['Sun'].y
    z_sol[i] = ps['Sun'].z

    x_jup[i] = ps['jupiter'].x
    y_jup[i] = ps['jupiter'].y 
    z_jup[i] = ps['jupiter'].z
    a_jup[i] = ps['jupiter'].a
    e_jup[i] = ps['jupiter'].e
    i_jup[i] = ps['jupiter'].inc
    pmjup[i] = ps['jupiter'].pomega
    lmjup[i] = ps['jupiter'].l

    for moon in range(num_L4):
        a_vals[moon,i] = ps['L4 {0}'.format(moon)].a
        e_vals[moon,i] = ps['L4 {0}'.format(moon)].e
        i_vals[moon,i] = ps['L4 {0}'.format(moon)].inc
        lmvals[moon,i] = ps['L4 {0}'.format(moon)].l
        pmvals[moon,i] = ps['L4 {0}'.format(moon)].pomega
        x_vals[moon,i] = ps['L4 {0}'.format(moon)].x
        y_vals[moon,i] = ps['L4 {0}'.format(moon)].y

    for moon in range(num_L5):
        a_vals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].a
        e_vals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].e
        i_vals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].inc
        lmvals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].l
        pmvals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].pomega
        x_vals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].x
        y_vals[moon + num_L4,i] = ps['L5 {0}'.format(moon)].y

i_vals/= radeg
i_jup /= radeg

troj_data = np.array((a_vals, e_vals, i_vals, pmvals, lmvals, x_vals, y_vals))
plnt_data = np.array((a_jup, e_jup, i_jup, pmjup, lmjup, x_jup, y_jup, z_jup))
star_data = np.array((mass, x_sol, y_sol, z_sol))

np.save("1k_Trojan_data_C2.npy", troj_data)
np.save("1k_Planet_data_C2.npy", plnt_data)
np.save("1k_Star_data_C2.npy", star_data)
np.save("1k_Timesteps_C2.npy", times)