#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

target = 'Ctrl'
radeg = np.pi/180

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

def L45_nonnorm(xjup,yjup,xsun,ysun):
    phi_jup = np.arctan2(yjup,xjup)
    
    phi_L4 = phi_jup + np.pi/3
    phi_L5 = phi_jup - np.pi/3
    
    xsep = (xsun - xjup)
    ysep = (ysun - yjup)
    
    r_jupsol = np.sqrt(xsep**2 + ysep**2)
    
    x_L4 = r_jupsol*np.cos(phi_L4)
    x_L5 = r_jupsol*np.cos(phi_L5)
    y_L4 = r_jupsol*np.sin(phi_L4)
    y_L5 = r_jupsol*np.sin(phi_L5)
    
    return np.array([x_L4,x_L5]), np.array([y_L4,y_L5])

def hill(a,e,m,M):
    return a*(1-e)*np.power(m/(3*M),1/3)


def r_pol(r,psi,M1,M2,a):
    q = M2/M1
    z = np.zeros((len(psi),len(r)))
    for i, phi in enumerate(psi):
        x_ = r*np.cos(phi)
        y_ = r*np.sin(phi)
        x = x_/a
        y = y_/a
        s1 = np.sqrt(x**2 + y**2)
        s2 = np.sqrt((x-1)**2 + y**2)
    
        term1 = 2/(s1*(1+q))
        term2 = 2*q/(s2*(1+q))
        term3 = (x - q/(1+q))**2
        term4 = y**2
        z[i] = term1 + term2 + term3 + term4
    return z

ast_d = np.load('{0}_Trojandata.npy'.format(target))
num_asts = len(ast_d[0,:,0])
print(ast_d.shape)

jup_d = np.load('{0}_Planetdata.npy'.format(target))
sol_d = np.load('{0}_Stardata.npy'.format(target))
times = np.load('{0}_Timesteps.npy'.format(target))

ast_a = ast_d[0]; ast_e = ast_d[1]; ast_i = ast_d[2] 
ast_o = ast_d[3]; ast_p = ast_d[4]; ast_l = ast_d[5]
ast_x = ast_d[6]; ast_y = ast_d[7]; ast_z = ast_d[8]
ast_meda = np.median(ast_a,axis=0)

jup_a = jup_d[0]; jup_e = jup_d[1]; jup_i = jup_d[2]; jup_p = jup_d[3]
jup_l = jup_d[4]; jup_x = jup_d[5]; jup_y = jup_d[6]; jup_z = jup_d[7]
sol_m = np.ones(100000); sol_l = np.ones(100000); sol_x = sol_d[2]; sol_y = sol_d[3]; sol_z = sol_d[4]
jhill = hill(jup_a,jup_e,9.546e-4,sol_m)
dst_jall = np.sqrt((ast_x - jup_x)**2 + (ast_y - jup_y)**2)

L45x, L45y = L45_nonnorm(jup_x,jup_y,sol_x,sol_y)
L4_xs = L45x[0]; L4_ys = L45y[0]
L5_xs = L45x[1]; L5_ys = L45y[1]

i_dif = np.zeros_like(ast_i)
i_int = ast_i[:,0]
for i in range(len(ast_a[0,:])):
    i_dif[:,i] = ast_i[:,i] - i_int
    
phi_vals = np.linspace(-np.pi,np.pi,500)
Z = r_pol(jup_a,phi_vals,sol_m,9.546e-4,jup_a)
Pot = np.flip(Z,1)

ast_r, ast_h = cart_to_pol(ast_x,ast_y)
jup_r, jup_h = cart_to_pol(jup_x,jup_y)
phdif = np.zeros_like(ast_h)
for i in range(len(jup_h)):
    phdif[:,i] = ast_h[:,i] - jup_h[i]
    
id4 = []
id5 = []
for i in range(num_asts):
    for it in range(len(jup_h)):
        if phdif[i,it] < -np.pi:
            phdif[i,it] = phdif[i,it] + 2*np.pi
        if phdif[i,it] > np.pi:
            phdif[i,it] = phdif[i,it] - 2*np.pi
    if phdif[i,0] > 0:
        id4.append(i)
    if phdif[i,0] < 0:
        id5.append(i)
        
print('Percentage at L4: %2.1f' %(len(id4)*100/num_asts))

liba = np.zeros((num_asts,200))
libp = np.zeros((num_asts,200))
for i in range(num_asts):
    for n in range(200):
        high = int(500*(n+1))
        loww = int(500*n)
        pmax = np.amax(phdif[i,loww:high])
        pmin = np.amin(phdif[i,loww:high])
        amax = np.amax(ast_a[i,loww:high])
        amin = np.amin(ast_a[i,loww:high])
        amid = np.median(jup_a[loww:high])
        
        if pmax > 0:
            mid = np.pi/3
        if pmax < 0:
            mid = -np.pi/3
            
        lip = ((pmax - mid) + (pmin - mid)) / 2
        lia = ((amax - amid)+(amin - amid)) / 2
        libp[i,n] = abs(lip)
        liba[i,n] = abs(lia)
        
indices = []
hillers = []
for i in range(num_asts):
    it = 0
    while it < len(ast_meda):
        a_focus = ast_a[i,it]
        a_media = ast_meda[it]
        if a_focus > a_media + 2:
            indices.append(i)
            break
        elif a_focus < a_media - 2:
            indices.append(i)
            break
        else:
            it += 1
    it = 0
    while it < len(jhill):
        d = dst_jall[i,it]
        h = jhill[it]
        if d <= h + 0.1:
            hillers.append(i)
            break
        else:
            it += 1

idx = np.array(indices)
hdx = np.array(hillers)

hill_not_sma = np.array(list(set(hillers) - set(indices)))
ndx = np.array(list(set(range(num_asts)) - set(indices)))

print("Number of escapers:            ", len(indices))
print("Number of hill crossers:       ", len(hillers))
pct = len(indices)/num_asts
print('Pct escaped / Total Asts:     %0.2f' %pct)

nrm_a = ast_a[ndx]; nrm_e = ast_e[ndx]; nrm_i = ast_i[ndx]; ndifi = i_dif[ndx]; nrmla = liba[ndx]
nrm_p = ast_p[ndx]; nrm_l = ast_l[ndx]; nrm_x = ast_x[ndx]; nrm_y = ast_y[ndx]; nrmlp = libp[ndx]


odd_a = ast_a[idx]; odd_e = ast_e[idx]; odd_i = ast_i[idx]; odifi = i_dif[idx]; oddla = liba[idx]
odd_p = ast_p[idx]; odd_l = ast_l[idx]; odd_x = ast_x[idx]; odd_y = ast_y[idx]; oddlp = libp[idx]

nrm_r, nrmph = cart_to_pol(nrm_x,nrm_y); odd_r, oddph = cart_to_pol(odd_x,odd_y)
jup_r, jupph = cart_to_pol(jup_x,jup_y); sol_r, solph = cart_to_pol(sol_x,sol_y)
L4_rs, L4phs = cart_to_pol(L4_xs,L4_ys); L5_rs, L5phs = cart_to_pol(L5_xs,L5_ys)

distj = np.sqrt((odd_x - jup_x)**2 + (odd_y - jup_y)**2)
disth = np.sqrt((ast_x[hdx] - jup_x)**2 + (ast_y[hdx] - jup_y)**2)
dists = np.sqrt((odd_x - sol_x)**2 + (odd_y - sol_y)**2)
jdist = np.sqrt((jup_x - sol_x)**2 + (jup_y - sol_y)**2)

earlies = []
laties = []
hill_cross = np.zeros(len(hdx))

for i in range(len(odd_a)):
    it = 0
    while it < 100000:
        a_focus = odd_a[i,it]
        a_media = ast_meda[it]
        if a_focus > a_media + 2:
            if it < 33333:
                earlies.append(i)
                break
            elif it > 70000:
                laties.append(i)
                break
            else:
                break
        elif a_focus < a_media - 2:
            if it < 33333:
                earlies.append(i)
                break
            elif it > 70000:
                laties.append(i)
                break
            else:
                break
        else:
            it += 1
            
for i in range(len(hdx)):
    it = 0
    while it < 100000:
        d = disth[i,it]
        h = jhill[it]
        if d <= h:
            hill_cross[i] = it
            break
        else:
            it += 1
            
horses = []
for number,n in enumerate(idx):
    i = 0
    while i < 5000:
        val = phdif[n,i]
        if 170*radeg <= val:
            horses.append(n)
            break
        elif val <= -170*radeg:
            horses.append(n)
            break
        elif -5*radeg <= val <= 5*radeg:
            horses.append(n)
            break
        i += 1
        
hrs = np.array(horses)
trs = np.array( list( set(idx) - set(horses) ) )
                        
edx = np.array(earlies)
ldx = np.array(laties)

print("Number of early escapees:       ", len(earlies), "  (escaped before .67 Myr)")
print("Number of late escapees:        ", len(laties), "   (escaped after %1.2f Myr)" %(times[70000]/1e6))
pct_e = len(earlies)/len(indices)
pct_l = len(laties)/len(indices)
print('Number early / Total escapees:   %0.2f' %pct_e)
print('Number late / Total escapees:    %0.2f' %pct_l)
pcT_e = len(earlies)/num_asts
pcT_l = len(laties)/num_asts
print('Number early / Total Asts.:      %0.2f' %pcT_e)
print('Number late / Total Asts.:       %0.2f' %pcT_l)


x_axis = np.linspace(0,times[33333]/1e6)
x_axi2 = np.linspace(times[70000]/1e6,times[-1]/1e6)

fig, ax = plt.subplots(3,figsize=(14,13),sharex=True,gridspec_kw={'height_ratios': [3, 1, .75]})
plt.subplots_adjust(hspace=0)

ax[0].plot(times/1e6,ast_meda,'k',lw=3)
ax[0].vlines([times[33333]/1e6,times[70000]/1e6],5,9.5,'b',alpha=0.8,zorder=0)

ax[0].fill_between(x_axis,5*np.ones_like(x_axis),9.5*np.ones_like(x_axis),facecolor='b',alpha=0.2,zorder=0)
ax[0].fill_between(x_axi2,5*np.ones_like(x_axis),9.5*np.ones_like(x_axis),facecolor='b',alpha=0.2,zorder=0)
ax[0].plot(times/1e6,jup_a,'gold',lw=3)
ax[0].legend(['Median Ast.','Planet'],fontsize=16,frameon=False,loc='upper left')
ax[0].set_ylabel('Semimajor Axis / AU',fontsize=16)
ax[0].set_ylim(5,9.5)
ax[0].set_xlim(0,2)
ax[0].text(0.18,7.25,"%1.i escaped" %len(earlies),fontsize=25)
ax[0].text(0.8,7.25,"%2.i escaped" %(len(indices) - len(earlies) - len(laties)),fontsize=25)
ax[0].text(1.48,7.25,"%2.i escaped" %len(laties),fontsize=25)

ax[1].plot(times/1e6,sol_l,'orange',lw=3,zorder=10)
ax[1].plot(times/1e6,sol_m,'g',ls=':',lw=3,zorder=10)
ax[1].vlines([times[33333]/1e6,times[70000]/1e6],0,4,'b',alpha=0.8,zorder=0)
ax[1].legend(["log Stellar Luminosity", "Stellar Mass"],fontsize=16,loc='center left',frameon=False)
ax[1].set_ylabel("Solar Units",fontsize=16)
ax[1].set_ylim(0,4)
ax[1].fill_between(x_axis,0*np.ones_like(x_axis),4*np.ones_like(x_axis),facecolor='b',alpha=0.2,zorder=0)
ax[1].fill_between(x_axi2,0*np.ones_like(x_axis),4*np.ones_like(x_axis),facecolor='b',alpha=0.2,zorder=0)
ax[1].set_xlabel('Time / Myr',fontsize=16)
ax[1].set_yticks([0,1,2,3])

ax[2].hist(hill_cross*20/1e6,edgecolor='k',facecolor='k',alpha=0.5,range=[0,2],bins=20)
ax[2].set_ylabel("Escapes",fontsize=16)
ax[2].set_xlabel("Time / Myr",fontsize=16)
ax[2].set_ylim(0,35)
ax[2].set_yticks([0,10,20,30])
fig.savefig('{0}_Timeseries.pdf'.format(target),dpi=300)

############

hist, axh = plt.subplots(1,4,figsize=(20,5))

axh[0].hist(nrm_a[:,0],edgecolor='k',histtype='step',range=[4.95,5.45])
axh[0].hist(odd_a[:,0],facecolor='r',alpha=0.7,range=[4.95,5.45])
axh[0].set_xlabel("SMA (AU)",fontsize=16)
axh[0].set_xlim(4.95,5.45)

axh[1].hist(nrm_e[:,0],edgecolor='k',histtype='step',range=[0,.25])
axh[1].hist(odd_e[:,0],facecolor='r',alpha=0.7,range=[0,.25])
axh[1].set_xlabel("Eccentricity",fontsize=16)
axh[1].set_xlim(0,0.25)

axh[2].hist(abs(nrmla[:,0]),edgecolor='k',histtype='step',range=[0,0.02],bins=20)
axh[2].hist(abs(liba[trs,0]),facecolor='r',alpha=0.7,range=[0,0.02],bins=20)
axh[2].set_xlabel("SMA Libration Amp. (AU)",fontsize=16)
axh[2].set_xlim(0,.02)
axh[2].set_xticks([0,0.005,0.01,0.015,0.02])

radeg = np.pi/180
axh[3].hist(abs(nrmlp[:,0])/radeg,edgecolor='k',histtype='step',range=[0,35])
axh[3].hist(abs(libp[trs,0])/radeg,facecolor='r',alpha=0.7,range=[0,35])
axh[3].set_xlabel(r"$\lambda$ Libration Amplitude (Deg.)",fontsize=16)
axh[3].set_xlim(0,35)
axh[3].legend(labels=['Stable','Escaped'],fontsize=14,frameon=False,loc='upper right')

hist.suptitle('Initial conditions',fontsize=18)
hist.savefig('{0}_Histograms.pdf'.format(target),dpi=300)

#############

orf, ora = plt.subplots(1,2,figsize=(15,5),gridspec_kw={'width_ratios': [2, 1]})
for i in range(len(ndx)):
    ora[0].plot(phdif[ndx[i],:500],ast_a[ndx[i],:500]/5.2,'k',alpha=0.01,zorder=5)
for i,tr in enumerate(trs):
    ora[0].plot(phdif[tr,:500],ast_a[tr,:500]/5.2,'r',alpha=0.05,zorder=10)
ora[0].set_xlim(-np.pi,np.pi)
ora[0].set_ylim(.9,1.1)
ora[0].set_xlabel(r"$\phi - \phi_{jup}$",fontsize=16)
ora[0].set_ylabel(r"SMA / $a_{jup}$",fontsize=16)
ora[0].vlines([-np.pi/3,np.pi/3],0.9,1.1,ls='--',zorder=0)
ora[0].set_xticks([-np.pi,-np.pi/2,-np.pi/3,0,np.pi/3,np.pi/2,np.pi])
ora[0].set_xticklabels([r"-$\pi$",r"-$\pi$/2",r"$L_5$",'0',r"$L_4$",r"$\pi$/2",r"$\pi$"])

sns.kdeplot(abs(nrmlp[:,0])/radeg,nrmla[:,0],shade=True,shade_lowest=None,cmap='Greys',levels=5,alpha=0.5)
sns.kdeplot(abs(libp[trs,0])/radeg,liba[trs,0],shade=True,shade_lowest=None,cmap='Reds',levels=5,alpha=0.5)
ora[1].set_ylabel("Init. SMA Libration (AU)",fontsize=16)
ora[1].set_xlabel(r"Init. $\lambda$ Libration (Deg.)",fontsize=16)
ora[1].set_xlim(0,35)
orf.tight_layout()
orf.savefig('{0}_Orbits.pdf'.format(target),dpi=300)

#############

norm = mpl.colors.Normalize(vmin = np.min(.005), vmax = np.max(.015), clip = False)

tim, tax = plt.subplots(figsize=(7,6))
scatter = tax.scatter(abs(libp[hdx,0])/radeg,hill_cross*20/1e6,c=abs(liba[hdx,0]),cmap='Reds',norm=norm)
tax.set_xlim(0,35)
tax.set_xlabel(r"Initial $\lambda$ Libration (Deg.)",fontsize=16)
tax.set_ylabel('Time of Encounter (Myr)',fontsize=16)
tim.colorbar(scatter, label='Initial SMA Libration (AU)')
tax.set_ylim(0,2)
tim.savefig('{0}_Eject_Perts.pdf'.format(target),dpi=300)

######################

hill_data = np.array((hdx,hill_cross))
np.save('{0}_Ejects.npy'.format(target), idx)
np.save('{0}_Hillcr.npy'.format(target), hill_data)