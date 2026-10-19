# MFBridge

Code for the experiments of the MFBridge paper.

| Folder | Contents |
| --- | --- |
| Synt-Reg/, Synt-Clas/ | Synthetic data experiments (regression / classification) |
| Real-Reg/, Real-Clas/ | Real-world data experiments (regression / classification) |
| Time-Reg/, Time-Cls/ | Computation time experiments (regression / classification) |

Requirements: Python 3, PyTorch, NumPy, SciPy, scikit-learn, matplotlib, mpi4py and an MPI runtime (e.g. OpenMPI). The real-world plots use LaTeX (usetex=True).


## Preparation (real-world data)

1. Unzip Datasets.zip in Real-Reg/ and in Real-Clas/ so that Real-Reg/Datasets/ and Real-Clas/Datasets/ exist.
2. Some datasets are too large to be included in this repository (GitHub file size limit). Download them separately and put them in Real-Reg/Datasets/ as follows.

Download<br>
pbc.csv, pharynx.csv, poker.csv, superconductivty.csv<br>
to Real-Reg/Datasets/ from URLs in Table 3.

These files are large and exceed GitHub's file size limit.

Change their categorical variables by "=MATCH(:,UNIQUE(:),0)-1" in Excel function,<br>
and set<br>
X = D, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17,<br>
Y = target<br>
for pbc.csv,<br>
X = Inst, sex, Treatment, Grade, Age, Condition, Site, T, N, Status,<br>
Y = target<br>
for pharynx.csv,<br>
X = att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10,<br>
Y = target<br>
for poker.csv, and<br>
X = number_of_elements, mean_atomic_mass, wtd_mean_atomic_mass, gmean_atomic_mass, wtd_gmean_atomic_mass, entropy_atomic_mass, wtd_entropy_atomic_mass, range_atomic_mass, wtd_range_atomic_mass, std_atomic_mass, wtd_std_atomic_mass, mean_fie, wtd_mean_fie, gmean_fie, wtd_gmean_fie, entropy_fie, wtd_entropy_fie, range_fie, wtd_range_fie, std_fie, wtd_std_fie, mean_atomic_radius, wtd_mean_atomic_radius, gmean_atomic_radius, wtd_gmean_atomic_radius, entropy_atomic_radius, wtd_entropy_atomic_radius, range_atomic_radius, wtd_range_atomic_radius, std_atomic_radius, wtd_std_atomic_radius, mean_Density, wtd_mean_Density, gmean_Density, wtd_gmean_Density, entropy_Density, wtd_entropy_Density, range_Density, wtd_range_Density, std_Density, wtd_std_Density, mean_ElectronAffinity, wtd_mean_ElectronAffinity, gmean_ElectronAffinity, wtd_gmean_ElectronAffinity, entropy_ElectronAffinity, wtd_entropy_ElectronAffinity, range_ElectronAffinity, wtd_range_ElectronAffinity, std_ElectronAffinity, wtd_std_ElectronAffinity, mean_FusionHeat, wtd_mean_FusionHeat, gmean_FusionHeat, wtd_gmean_FusionHeat, entropy_FusionHeat, wtd_entropy_FusionHeat, range_FusionHeat, wtd_range_FusionHeat, std_FusionHeat, wtd_std_FusionHeat, mean_ThermalConductivity, wtd_mean_ThermalConductivity, gmean_ThermalConductivity, wtd_gmean_ThermalConductivity, entropy_ThermalConductivity, wtd_entropy_ThermalConductivity, range_ThermalConductivity, wtd_range_ThermalConductivity, std_ThermalConductivity, wtd_std_ThermalConductivity, mean_Valence, wtd_mean_Valence, gmean_Valence, wtd_gmean_Valence, entropy_Valence, wtd_entropy_Valence, range_Valence, wtd_range_Valence, std_Valence, wtd_std_Valence,<br>
Y = critical_temp<br>
for superconductivty.csv.


## Synthetic Data Experiments

Run in Synt-Reg/ or Synt-Clas/:

```
source bash.sh
```

This runs all methods with mpiexec (16 processes), summarizes the results, and creates the figures.


## Real-World Data Experiments

Run in Real-Reg/Codes/ or Real-Clas/Codes/:

```
source bash.sh
```

This runs all methods on all datasets with mpiexec (16 processes), then runs summary1.py, summary2.py and plot1.py.
Results are saved in Results/ (summaries in Results/summary/). Delete Results/summary/ before re-running summaries, because existing summary files are skipped.


## Computation Time Experiments

Run in Time-Reg/ or Time-Cls/:

```
python bash.py
python table1.py
```

bash.py measures the training time of each method (100 seeds, dim in {10, 20, 40}, M in {10, 20, 40}) and saves it in Results/.
table1.py prints the table (mean and standard deviation) in LaTeX format.
bash.py uses GPU 0 (CUDA = [0]); set it to [-1] to run on CPU.
