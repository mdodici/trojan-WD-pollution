#!/usr/bin/env python
# coding: utf-8

# In[8]:


import rebound
import reboundx
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


###############
### IMPORTS ###
###############


params = np.load('sample_params.npy')

file = np.loadtxt('1M_track.txt')
sol_t = file[807:,0]
sol_m = file[807:,1]
sol_l = file[807:,6]
log_l = InterpolatedUnivariateSpline(sol_t, sol_l,k=1)
m_sol = InterpolatedUnivariateSpline(sol_t, sol_m,k=1)


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

def yark(simp, rebx_force, particles, N):
    sim = simp.contents
    part = sim.particles
    
    current_time = sim.t + T0
    L_sol = np.power(10,log_l(current_time))*0.000235 # solar luminosity in au^2 M_sol/yr^3
    
    sim.move_to_hel()
    for troj in range(num_tr):
    
        i = troj + 1
        
        x = part[i].x  ; y = part[i].y  ; z = part[i].z
        R = troj_radii[i-1]
        m_ast = troj_masses[i-1]
    
        c = 63197.8 # speed of light in au/yr
        r = np.sqrt(x**2 + y**2 + z**2)
        A = (R**2 * L_sol)/(4*m_ast*c)
    
        part[i].ax += (A/r**3) * (0.25*y + x)
        part[i].ay += (A/r**3) * (0.25*x + y)
        part[i].az += (A/r**3) * z
    return


############################
############################
############################

###      SIMULATION      ###

############################
############################
############################


T0 = sol_t[0]
t_tot = 2000000
Nout = 100000
times = np.linspace(0,t_tot,Nout)

M0 = m_sol(T0)
num_tr = len(params[0])

sim = rebound.Simulation()

sim.add(m=M0,x=0, y=0, z=0, vx=0, vy=0, vz=0, hash='Sun')
add_tr(sim, params)
sim.add(m=9.543e-4, a=5.2, e=.04839, inc=.022689, Omega=0, omega=0, hash='jupiter')

sim.integrator = 'whfast'
sim.dt = 0.5
sim.move_to_com()

ps = sim.particles


################
###  Extras  ###
################


rebx = reboundx.Extras(sim)

yrkv = rebx.create_force("yarkovsky")
yrkv.force_type = "pos"
yrkv.update_accelerations = yark
rebx.add_force(yrkv)

rad_ast = 10                                         # radius in km
troj_radii = np.full(num_tr, rad_ast/1.496e+8)      # gives each asteroid a radius in AU
mass_typic = 3*(4/3)*np.pi*(rad_ast*100000)**3      # gives typical mass @ this radius, w/ density = 3 g cm^-3
troj_masses = np.full(num_tr,mass_typic)            # gives array of values of that mass
troj_masses /= 1.9891e33                            # divides each mass by M_sol to get masses in M_sol


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

    ps[0].m = m_sol(sim.t + T0)
    sim.move_to_com()

    lsol[i] = log_l(sim.t + T0)    
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

np.save("10kC_Trojandata.npy", troj_data)
np.save("10kC_Planetdata.npy", plnt_data)
np.save("10kC_Stardata.npy", star_data)
np.save("10kC_Timesteps.npy", times)

