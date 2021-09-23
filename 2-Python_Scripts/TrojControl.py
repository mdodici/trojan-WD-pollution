#!/usr/bin/env python
# coding: utf-8

# In[8]:


import rebound
import numpy as np


###############
### IMPORTS ###
###############


params = np.load('sample_params.npy')


###################
### DEFINITIONS ###
###################


radeg = np.pi/180

def add_tr(sim, pars):
    a = pars[0]
    e = pars[1]
    c = pars[2]*radeg
    p = pars[3]
    l = pars[4] 

    for i in range(len(a)):
        sem = a[i]
        ecc = e[i]
        icl = c[i]
        pme = p[i]
        lam = l[i]
        has = 'tr_{0}'.format(i)
        sim.add(m=0, primary=sim.particles['Sun'], a=sem, e=ecc, inc=icl, pomega=pme, l=lam, hash=has)


############################
############################
############################

###      SIMULATION      ###

############################
############################
############################


t_tot = 2000000
Nout = 100000
times = np.linspace(0,t_tot,Nout)

M0 = 1
num_tr = len(params[0])

sim = rebound.Simulation()

sim.add(m=M0,x=0, y=0, z=0, vx=0, vy=0, vz=0, hash='Sun')
add_tr(sim, params)
sim.add(m=9.543e-4, a=5.2, e=.04839, inc=.022689, Omega=0, omega=0, hash='jupiter')

sim.integrator = 'whfast'
sim.dt = 0.5
sim.move_to_com()

ps = sim.particles


#########################################
##  Parameter tracking initialization  ##
#########################################


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
omvals = np.zeros((num_tr, Nout))
pmvals = np.zeros((num_tr, Nout))
lmvals = np.zeros((num_tr, Nout))

x_vals = np.zeros((num_tr, Nout))
y_vals = np.zeros((num_tr, Nout))
z_vals = np.zeros((num_tr, Nout))

for moon in range(num_tr):
    a_vals[moon,0] = ps['tr_{0}'.format(moon)].a
    e_vals[moon,0] = ps['tr_{0}'.format(moon)].e
    i_vals[moon,0] = ps['tr_{0}'.format(moon)].inc
    lmvals[moon,0] = ps['tr_{0}'.format(moon)].l
    omvals[moon,0] = ps['tr_{0}'.format(moon)].Omega
    pmvals[moon,0] = ps['tr_{0}'.format(moon)].pomega
    x_vals[moon,0] = ps['tr_{0}'.format(moon)].x
    y_vals[moon,0] = ps['tr_{0}'.format(moon)].y
    z_vals[moon,0] = ps['tr_{0}'.format(moon)].z


###########################
###########################
###########################

####      RUNNING      ####

###########################
###########################
###########################

for i, time in enumerate(times):
    sim.integrate(time)
    sim.move_to_com()

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

    for moon in range(num_tr):
        a_vals[moon,i] = ps['tr_{0}'.format(moon)].a
        e_vals[moon,i] = ps['tr_{0}'.format(moon)].e
        i_vals[moon,i] = ps['tr_{0}'.format(moon)].inc
        lmvals[moon,i] = ps['tr_{0}'.format(moon)].l
        omvals[moon,i] = ps['tr_{0}'.format(moon)].Omega
        pmvals[moon,i] = ps['tr_{0}'.format(moon)].pomega
        x_vals[moon,i] = ps['tr_{0}'.format(moon)].x
        y_vals[moon,i] = ps['tr_{0}'.format(moon)].y
        z_vals[moon,i] = ps['tr_{0}'.format(moon)].z


##############
##  Saving  ##
##############


i_vals/= radeg
i_jup /= radeg

troj_data = np.array((a_vals, e_vals, i_vals, omvals, pmvals, lmvals, x_vals, y_vals, z_vals))
plnt_data = np.array((a_jup, e_jup, i_jup, pmjup, lmjup, x_jup, y_jup, z_jup))
star_data = np.array((mass, lsol, x_sol, y_sol, z_sol))

np.save("Ctrl_Trojandata.npy", troj_data)
np.save("Ctrl_Planetdata.npy", plnt_data)
np.save("Ctrl_Stardata.npy", star_data)
np.save("Ctrl_Timesteps.npy", times)

