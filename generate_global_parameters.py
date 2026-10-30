import os
import numpy as np
import pandas as pd



##### Build vaccine efficacy data #####


# Interpret vaccine data and manipulate into relevant parameters
def build_vaccine_efficacy(r, data_path):

    eff_df = pd.read_csv(
        os.path.join(data_path, f'global_parameters/efficacies_{r}.csv'),
        usecols=range(1,5), dtype=np.float64
    )

    VE_i = eff_df['Infection_eff']
    VE_s = eff_df['Symptom_eff']
    VE_h = eff_df['Hosp_eff']
    VE_d = eff_df['Death_eff']

    # Changing cumulative proportions into relative proportions
    VE_d = np.divide(VE_d-VE_h, 1-VE_h)
    ##### Questionable? should use VE_s here
    VE_h = np.divide(VE_h-VE_i, 1-VE_i)
    VE_s = np.divide(VE_s-VE_i, 1-VE_i)

    ##### Unused in this model, how vaccines affect transmission
    regimes_nu_tra = [1] * 6 

    regimes_mu_symp = np.nan_to_num(1 - VE_s).tolist()
    regimes_nu_inf = np.nan_to_num(1 - VE_i).tolist()
    regimes_rho_sev_h = np.nan_to_num(1 - VE_h).tolist()
    regimes_phi_sev_d = np.nan_to_num(1 - VE_d).tolist()

    # Conerting into dictionary as standard
    vaccine_status = ["U", "F", "B", "W1", "W2"]
    mu_regime = dict(zip(vaccine_status, regimes_mu_symp[:5]))
    nu_regime = dict(zip(vaccine_status, regimes_nu_inf[:5]))
    rho_regime = dict(zip(vaccine_status, regimes_rho_sev_h[:5]))
    phi_regime = dict(zip(vaccine_status, regimes_phi_sev_d[:5]))

    # Return 4 dicts[vaccine_status][1]
    return mu_regime, nu_regime, rho_regime, phi_regime


##### #####



##### Build remaining global parameters #####


# Interpret remaining data to relevant form
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

    # Return data types found in S1_Table
    return regimes_omega, regimes_alpha, regimes_gamma, regimes_tau, regimes_w


##### #####