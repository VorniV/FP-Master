import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#------------------------------ Heizrate ------------------------------
data1 = pd.read_csv("Dipolrelaxation_Messung_1.txt", sep="\t")
data2 = pd.read_csv("Dipolrelaxation_Messung_2.txt", sep="\t")

print(data1.iloc[len(data1)-1,0])

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


#------------------------------ Fits ------------------------------

plt.plot(data1.iloc[:,0], data1.iloc[:,1], linestyle=":")
plt.xlabel("T [K]")
plt.ylabel("I [pA]")
plt.savefig("plot1.pdf")
