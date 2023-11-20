import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy import integrate


#------------------------------ Heizrate ------------------------------
data1 = pd.read_csv("Dipolrelaxation_Messung_1.txt", sep="\t")
data2 = pd.read_csv("Dipolrelaxation_Messung_2.txt", sep="\t")

data1["#Temperatur[C]"] = data1["#Temperatur[C]"] + 273.15
data2["#Temperatur[C]"] = data2["#Temperatur[C]"] + 273.15

data1["#Strom[pA]"] = data1["#Strom[pA]"] * 10**(-11)
data2["#Strom[pA]"] = data2["#Strom[pA]"] * 10**(-11)

#print(data1.iloc[len(data1)-1,0])

#heat1 = (data1.iloc[len(data1)-1,0] - data1.iloc[0,0]) / (len(data1) - 1)
#heat2 = (data2.iloc[len(data2)-1,0] - data2.iloc[0,0]) / (len(data2) - 1)


heat1_list=[]
for n in range(len(data1)-1):
    heat1_list.append( data1.iloc[n+1,0] - data1.iloc[n,0] )

heat1_pd = pd.DataFrame(heat1_list)
heat1 = heat1_pd.mean()
heat1_std = heat1_pd.std()

heat2_list=[]
for n in range(len(data2)-1):
    heat2_list.append( data2.iloc[n+1,0] - data2.iloc[n,0] )

heat2_pd = pd.DataFrame(heat2_list)
heat2 = heat2_pd.mean()
heat2_std = heat2_pd.std()


#------------------------------ Untergrund Fit ------------------------------

def exp(T,A,b,C):
    return A * np.exp(-b/T) +C

head1 = 11
head_min1 = 0
tail1 = 11
tail_min1 = 22

head2 = 11
head_min2 = 0
tail2 = 9
tail_min2 = 16

fit_points1 = pd.concat([data1.iloc[head_min1 : head_min1+head1 , :]  ,  data1.iloc[len(data1)-(tail1+tail_min1):len(data1)-tail_min1 , :]], axis=0)
fit_points2 = pd.concat([data2.iloc[head_min2 : head_min2+head2 , :]  ,  data2.iloc[len(data2)-(tail2+tail_min2):len(data2)-tail_min2 , :]], axis=0)

par1, cov1 = curve_fit(exp, fit_points1.iloc[:,0], fit_points1.iloc[:,1], p0=[0.2*10**(12), 6000, -5])
par2, cov2 = curve_fit(exp, fit_points2.iloc[:,0], fit_points2.iloc[:,1], p0=[0.2*10**(12), 6000, -5])

print(par1)
print(par2)

T1 = np.linspace(210,300,10000)
T2 = np.linspace(210,300,10000)

plt.plot(data1.iloc[:,0], data1.iloc[:,1], ".")
plt.plot(fit_points1.iloc[:,0],fit_points1.iloc[:,1], "rx")
plt.plot(T1, exp(T1, *par1))
plt.xlabel("T [K]")
plt.ylabel("I [pA]")
plt.savefig("plot1.pdf")
plt.close()


plt.plot(data2.iloc[:,0], data2.iloc[:,1], ".")
plt.plot(fit_points2.iloc[:,0],fit_points2.iloc[:,1], "rx")
plt.plot(T2, exp(T2, *par2))
plt.xlabel("T [K]")
plt.ylabel("I [pA]")
plt.savefig("plot2.pdf")
plt.close()


#------------------------------ Signal plot ------------------------------

sig1_list=[]
for n in range(len(data1)):
    sig1_list.append( data1.iloc[n,1] - exp(data1.iloc[n,0], *par1) )

sig1_pd = pd.DataFrame(sig1_list)


sig2_list=[]
for n in range(len(data2)):
    sig2_list.append( data2.iloc[n,1] - exp(data2.iloc[n,0], *par2) )

sig2_pd = pd.DataFrame(sig2_list)

sig1_max = sig1_pd.idxmax()
sig2_max = sig2_pd.idxmax()


plt.plot(data1.iloc[:,0], sig1_pd, ".")
plt.plot(data1.iloc[0:head1,0] , sig1_pd.iloc[0:head1] , "rx", label="Untergrund")
plt.plot(data1.iloc[(len(data1)- (tail_min1 + tail1) ):len(data1),0], sig1_pd.iloc[(len(data1)- (tail_min1 + tail1) ):len(data1),0], "rx")
plt.plot(data1.iloc[head1:sig1_max[0]+1, 0], sig1_pd.iloc[head1:sig1_max[0]+1, 0], "bx", label="Polarisations-Auswertung")
plt.plot(data1.iloc[sig1_max[0]+1:len(data1)- (tail_min1 + tail1), 0], sig1_pd.iloc[sig1_max[0]+1:len(data1)- (tail_min1 + tail1), 0],"gx", label="Integrale-Auswertung")
plt.xlabel("T [K]")
plt.ylabel("I [pA]")
plt.legend()
plt.savefig("sig1.pdf")
plt.close()


plt.plot(data2.iloc[:,0], sig2_pd, ".")
plt.plot(data2.iloc[0:head2,0] , sig2_pd.iloc[0:head2] , "rx", label="Untergrund")
plt.plot(data2.iloc[(len(data2)- (tail_min2 + tail2) ):len(data2),0], sig2_pd.iloc[(len(data2)- (tail_min2 + tail2) ):len(data2),0], "rx")
plt.plot(data2.iloc[head2:sig2_max[0]+1, 0], sig2_pd.iloc[head2:sig2_max[0]+1, 0], "bx", label="Polarisations-Auswertung")
plt.plot(data2.iloc[sig2_max[0]+1:len(data2)- (tail_min2 + tail2), 0], sig2_pd.iloc[sig2_max[0]+1:len(data2)- (tail_min2 + tail2), 0],"gx", label="Integrale-Auswertung")
plt.xlabel("T [K]")
plt.ylabel("I [pA]")
plt.legend()
plt.savefig("sig2.pdf")
plt.close()


#------------------------------ Polarisationsansatz ------------------------------

#W:
def pol(T,m,n):
    return m * T + n


param1, cova1 = curve_fit(pol, 1 / data1.iloc[head1:sig1_max[0]+1, 0],  np.log(sig1_pd.iloc[head1:sig1_max[0]+1, 0]))
param2, cova2 = curve_fit(pol, 1 / data2.iloc[head2:sig2_max[0]+1, 0],  np.log(sig2_pd.iloc[head2:sig2_max[0]+1, 0]))

cova1 = np.sqrt(np.diag(cova1))
cova2 = np.sqrt(np.diag(cova2))

uparam1 = unp.uarray(param1, cova1)
uparam2 = unp.uarray(param2, cova2)


x1 = np.linspace(1/data1.iloc[head1, 0],1/data1.iloc[sig1_max[0]+1, 0], 10000)
x2 = np.linspace(1/data2.iloc[head2, 0],1/data2.iloc[sig2_max[0]+1, 0], 10000)


plt.plot(1/data1.iloc[head1:sig1_max[0]+1, 0], np.log( sig1_pd.iloc[head1:sig1_max[0]+1, 0]), "x", label="Messwerte")
plt.plot(x1, pol(x1, *param1), label = "Ausgleichsgerade")
plt.legend()
plt.savefig("Polarisationsansatz1.pdf")
plt.close()

plt.plot(1/data2.iloc[head2:sig2_max[0]+1, 0], np.log( sig2_pd.iloc[head2:sig2_max[0]+1, 0]), "x", label="Messwerte")
plt.plot(x2, pol(x2, *param2), label = "Ausgleichsgerade")
plt.legend()
plt.savefig("Polarisationsansatz2.pdf")
plt.close()


W1 = const.k * -1 * uparam1[0]
W2 = const.k * -1 * uparam2[0]


W1 = W1 / const.e 
W2 = W2 / const.e 

print(W1)
print(W2)

#Tau:
tau1_max = (const.k * data1.iloc[sig1_max,0] **2) / (heat1[0] * W1 *const.e)
tau2_max = (const.k * data2.iloc[sig2_max,0] **2) / (heat2[0] * W2 *const.e)

tau1 = tau1_max * unp.exp((-W1*const.e) / (const.k * data1.iloc[sig1_max,0])) #* 10**(12)
tau2 = tau2_max * unp.exp(-W2*const.e / (const.k * data2.iloc[sig2_max,0])) #* 10**(12)

print("Tau_max: ", tau1_max, "\n")
print("Tau_max: ", tau2_max, "\n")
print("Tau: ",tau1, "\n")
print("Tau: ",tau2, "\n")


#------------------------------ Integralansatz ------------------------------

#W:
int1 = integrate.cumtrapz(sig1_pd.iloc[head1:len(data1)- (tail_min1 + tail1), 0], data1.iloc[head1:len(data1)- (tail_min1 + tail1), 0], initial= data1.iloc[head1, 0] )
int2 = integrate.cumtrapz(sig2_pd.iloc[head2:len(data2)- (tail_min2 + tail2), 0], data2.iloc[head2:len(data2)- (tail_min2 + tail2), 0], initial= data2.iloc[head2, 0] )

parame1, covar1 = curve_fit(pol, 1 / data1.iloc[head1:len(data1)- (tail_min1 + tail1), 0],  np.log(int1 / sig1_pd.iloc[head1:len(data1)- (tail_min1 + tail1), 0]))
parame2, covar2 = curve_fit(pol, 1 / data2.iloc[head2:len(data2)- (tail_min2 + tail2), 0],  np.log(int2 / sig2_pd.iloc[head2:len(data2)- (tail_min2 + tail2), 0]))

covar1 = np.sqrt(np.diag(covar1))
covar2 = np.sqrt(np.diag(covar2))

uparame1 = unp.uarray(parame1, covar1)
uparame2 = unp.uarray(parame2, covar2)

W1_I = const.k * -1 * uparame1[0]
W2_I = const.k * -1 * uparame2[0]


W1_I = W1_I / const.e 
W2_I = W2_I / const.e 

print(W1_I)
print(W2_I)

#Tau:
tau1_max_I = (const.k * data1.iloc[sig1_max,0] **2) / (heat1[0] * W1_I *const.e)
tau2_max_I = (const.k * data2.iloc[sig2_max,0] **2) / (heat2[0] * W2_I *const.e)

tau1_I = tau1_max_I * unp.exp((-W1_I*const.e) / (const.k * data1.iloc[sig1_max,0])) #* 10**(12)
tau2_I = tau2_max_I * unp.exp((-W2_I*const.e) / (const.k * data2.iloc[sig2_max,0])) #* 10**(12)

print("Tau_max: ", tau1_max_I, "\n")
print("Tau_max: ", tau2_max_I, "\n")
print("Tau: ",tau1_I, "\n")
print("Tau: ",tau2_I, "\n")


plt.plot(1/data1.iloc[head1:len(data1)- (tail_min1 + tail1), 0], np.log( sig1_pd.iloc[head1:len(data1)- (tail_min1 + tail1), 0]), "x", label="Messwerte")
plt.plot(x1, pol(x1, *param1), label = "Ausgleichsgerade")
plt.legend()
plt.savefig("Integralansatz1.pdf")
plt.close()

plt.plot(1/data2.iloc[head2:len(data2)- (tail_min2 + tail2), 0], np.log( sig2_pd.iloc[head2:len(data2)- (tail_min2 + tail2), 0]), "x", label="Messwerte")
plt.plot(x2, pol(x2, *param2), label = "Ausgleichsgerade")
plt.legend()
plt.savefig("Integralansatz2.pdf")
plt.close()
