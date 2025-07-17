import numpy as np
import attitude

# usage example of attitude.py

omgs=[]
acs= []
times = []
with open("C:\\Users\\Artur\\Downloads\\Telegram Desktop\\data16 - AMUR.txt") as f:
    next(f)
    for line in f:
        d = line.split(",")
        times.append(float(d[0]) / 1000)
        acs.append(np.array([float(d[1]),float(d[2]),float(d[3])]))
        omgs.append(np.array([float(d[4]),float(d[5]),float(d[6])]))
att = attitude.Attitude(0.007)
# getting delta times
dtimes=[]
for i in range(len(times)-1):
    dtimes.append(times[i+1]-times[i])
dtimes=dtimes + [dtimes[0]]
att.calculate(dtimes,acs,omgs) # performing calculations

def save(fnm,lst):
    f = open(fnm,"w")
    for elem in lst:
        s = ""
        for e in elem:
            s+=str(e)+","
        f.write(s+"\n")
    f.close()

# save("acc_test.txt", att.get_accs())
save("gs_test.txt",att.get_gs())
