import sys
import numpy as np

target = sys.argv[2]
path = sys.argv[1]
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

ast_d = np.load('{0}_Trojandata.npy'.format(path))
num_asts = len(ast_d[0,:,0])
print(ast_d.shape)

jup_d = np.load('{0}_Planetdata.npy'.format(path))
sol_d = np.load('{0}_Stardata.npy'.format(path))
times = np.load('{0}_Timesteps.npy'.format(path))

ast_a = ast_d[0]; ast_e = ast_d[1]; ast_i = ast_d[2] 
ast_o = ast_d[3]; ast_p = ast_d[4]; ast_l = ast_d[5]
ast_x = ast_d[6]; ast_y = ast_d[7]; ast_z = ast_d[8]
ast_meda = np.median(ast_a,axis=0)

jup_a = jup_d[0]; jup_e = jup_d[1]; jup_i = jup_d[2]; jup_p = jup_d[3]
jup_l = jup_d[4]; jup_x = jup_d[5]; jup_y = jup_d[6]; jup_z = jup_d[7]
sol_m = sol_d[0]; sol_l = sol_d[1]; sol_x = sol_d[2]; sol_y = sol_d[3]; sol_z = sol_d[4]
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

disth = np.sqrt((ast_x[hdx] - jup_x)**2 + (ast_y[hdx] - jup_y)**2)

hill_cross = np.zeros(len(hdx))        
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
                        
hill_data = np.array((hdx,hill_cross))
np.save('{0}_Ejects.npy'.format(target), idx)
np.save('{0}_Hillcr.npy'.format(target), hill_data)

reruns = np.zeros((9,len(idx)))
time_vals = np.zeros(len(idx))
no_hill = np.array(list(set(indices) - set(hillers)))
if len(no_hill) != 0:
    for i, ast in enumerate(no_hill):
        it = 0
        while it < 100000:
            if abs(ast_a[ast,it] - jup_a[it]) > 1:
                dw = int(it-1000)
                if dw < 0:
                    dw = 0
                reruns[:,i] = ast_d[:,ast,dw]
                time_vals[i] = dw
                break
            it += 1

for i, ast in enumerate(hdx):
    mid = hill_cross[i]
    dw = int(mid - 1000)
    if dw < 0:
        dw = 0
    reruns[:,i+len(no_hill)-1] = ast_d[:,ast,dw]
    time_vals[i+len(no_hill)-1] = dw

np.save('{0}_redopars.npy'.format(target), reruns)
np.save('{0}_redotime.npy'.format(target), time_vals)