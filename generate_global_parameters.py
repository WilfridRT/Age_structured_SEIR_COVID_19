import numpy as np
import pandas as pd
import os
from collections import defaultdict

nu_tra = []
mu_symp = []
nu_inf = []
rho_sev_h = []
phi_sev_d = []

omega = []
gamma = []
alpha = []
tau = []
w = []

def build_vaccine_efficacy(r, data_path):

    eff_df = pd.read_csv(
        os.path.join(data_path, f'global_parameters/efficacies_{r}.csv'),
        usecols=range(1,5), dtype=np.float64
    )

    VE_i = eff_df['Infection_eff']
    VE_s = eff_df['Symptom_eff']
    VE_h = eff_df['Hosp_eff']
    VE_d = eff_df['Death_eff']

    VE_d = np.divide(VE_d-VE_h, 1-VE_h)
    VE_h = np.divide(VE_h-VE_i, 1-VE_i)
    VE_s = np.divide(VE_s-VE_i, 1-VE_i)

    regimes_nu_tra = [1] * 6 # Unused in this model, how vaccines affect transmission
    regimes_mu_symp = np.nan_to_num(1 - VE_s).tolist()
    regimes_nu_inf = np.nan_to_num(1 - VE_i).tolist()
    regimes_rho_sev_h = np.nan_to_num(1 - VE_h).tolist()
    regimes_phi_sev_d = np.nan_to_num(1 - VE_d).tolist()

    vaccine_status = ["U", "F", "B", "W1", "W2"]
    mu_regime = dict(zip(vaccine_status, regimes_mu_symp[:5]))
    nu_regime = dict(zip(vaccine_status, regimes_nu_inf[:5]))
    rho_regime = dict(zip(vaccine_status, regimes_rho_sev_h[:5]))
    phi_regime = dict(zip(vaccine_status, regimes_phi_sev_d[:5]))

    nu_tra.append(regimes_nu_tra)
    mu_symp.append(regimes_mu_symp) # mu
    nu_inf.append(regimes_nu_inf) # nu
    rho_sev_h.append(regimes_rho_sev_h) # rho
    phi_sev_d.append(regimes_phi_sev_d) # phi

    return mu_regime, nu_regime, rho_regime, phi_regime

def build_global_parameters(r, data_path):

    param_df = pd.read_csv(
        os.path.join(data_path, f'global_parameters/parameters_{r}.csv'),
        dtype=np.float64
    )

    regimes_omega = param_df['transmission'].tolist()[0]
    regimes_alpha = 1
    regimes_gamma = param_df['recovery'].tolist()[0]
    regimes_tau = param_df['asymptomatic_transmission'].tolist()[0]
    regimes_w = param_df['waning_rate'].tolist()[0]

    omega.append(regimes_omega)
    alpha.append(regimes_alpha)
    gamma.append(regimes_gamma)
    tau.append(regimes_tau)
    w.append(regimes_w)

    return regimes_omega, regimes_alpha, regimes_gamma, regimes_tau, regimes_w

