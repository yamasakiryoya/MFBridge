#python exist-1.py -> exist-1.txt
import os
import itertools
import numpy as np

DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
N = 5

#"""
print(1)
for METH,DATA,SIZE,l in itertools.product(METHs,DATAs,[25,50,100,200,400],range(len(LAMs2))):
  LAM = LAMs2[l]
  for SEED in range(N):
    if not os.path.isfile("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,DATA,SIZE,LAM,SEED)):
      print(DATA,SIZE,LAM,SEED,METH)
#"""
#"""
print(2)
for METH,DATA,SIZE in itertools.product(METHs,DATAs,[25,50,100,200,400]):
  result = np.zeros((len(LAMs2),N))
  for SEED,l in itertools.product(range(N),range(len(LAMs2))):
    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,DATA,SIZE,LAMs2[l],SEED), delimiter=",")
    result[l,SEED] = np.min(tmp[:,-2])
  LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
  flag = 0
  for SEED,l in itertools.product(range(N),range(len(LAMs3))):
    if os.path.exists("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METH,DATA,SIZE,LAMs3[l],SEED))==False: flag = 1
  if flag==1: print(METH,DATA,SIZE)
#"""