import numpy as np
import attitude

omgs = []
acs = []
with open("C:\\Users\\Artur\\Downloads\\Telegram Desktop\\data16 - AMUR.txt") as f:
    next(f)
    for line in f:
        d = line.split(",")
        acs.append(np.array([float(d[1]), float(d[2]), float(d[3])]))
        omgs.append(np.array([float(d[4]), float(d[5]), float(d[6])]))
att = attitude.Attitude(0.007)

att.calculate(acs, omgs)


def save(fnm, lst):
    ff = open(fnm, "w")
    for elem in lst:
        s = ""
        for e in elem:
            s += str(e) + ","
        ff.write(s + "\n")
    ff.close()


save("acc_test.txt", att.get_accs())
"""# hello
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion

# q = pyquaternion.Quaternion(axis=[0,0,1],angle=0)
q = np.array([1,0,0,0])

qs = []
aas = []
def save(fnm,lst):
    f = open(fnm,"w")
    for elem in lst:
        s = ""
        for e in elem:
            s+=str(e)+","
        f.write(s+"\n")
    f.close()

print("loading file...")
flag=False
wx0=0
wy0=0
wz0=0
with open("data_cut.txt") as f:
    for line in f:
        d = line.split(",")
            
        wx = float(d[4]) - wx0
        wy = float(d[5]) - wy0
        wz = float(d[6]) - wz0
        if not flag:
            wx0 = wx
            wy0=wy
            wz0=wz
            flag = True
        aas.append(np.array([float(d[1]),float(d[2]),float(d[3])]))
        dt = 0.007 # TODO!!!! # TODO!!!! # TODO!!!!

        W = np.array([[2/dt,-wx,-wy,-wz], # taken from https://mariogc.com/post/angular-velocity-quaternions/
            [wx,2/dt,wz,-wy],
            [wy,-wz,2/dt,wx],
            [wz,wy,-wx,2/dt]])

        q = dt/2 * np.dot(W,q)
        q = q/np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2) # normalizing
        qs.append(pyquaternion.Quaternion(q))

print(f"Done, got {len(qs)} lines")
print("Sample series:")
for j in range(10):
    print(qs[len(qs)//2+j])
print("calculating initial angles...")
init_as = aas[0]
#TODO count initial angles
# Then we need to get list of gs in aircraft's coordinate system
g = init_as
g_quat = pyquaternion.Quaternion(0,-g[0],-g[1],-g[2])
gs = []
for k in qs:
    g_rot = k.inverse * g_quat * k
    # print(f"Vector: {g_rot.imaginary}, Module: {np.linalg.norm(g_rot.imaginary)}")
    gs.append(g_rot.imaginary)
save("qs.txt",qs)
print("Calculated gs in bound coordinate system")
print("Sample: ")
for j in range(10):
    print(gs[len(gs)//2+j])
print("calculating accelerations in earths coordinate system...")
for i in range(len(aas)):
    aas[i] = aas[i] + gs[i]
print("accelerations sample:")
for j in range(10):
    print(f"{aas[len(aas)//2+j]} Module:{np.linalg.norm(aas[len(aas)//2+j])}")
print("as saving in file as.txt...")
p = open("acc.txt","w")
for j in aas:
    p.write(f"{j[0]},{j[1]},{j[2]}\n")
print("done!")
"""
