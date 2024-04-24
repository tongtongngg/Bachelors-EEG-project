#%%
from mne.preprocessing import read_ica
import mne
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import pandas
from mne.preprocessing import ICA
from mne.stats import permutation_cluster_test
from mne.time_frequency import (
    AverageTFR,
    tfr_array_morlet,
    tfr_morlet,
    tfr_multitaper,
    tfr_stockwell,
)

#%%
#Import data
cleaned_epochs7 = mne.read_epochs('Data\cleaned_data_epochs.fif', preload=True)
ica7 = read_ica('Data\ICAsub7infomax-ica.fif')
ICAmotor7 = ica7.get_sources(cleaned_epochs7).pick("ICA007").rename_channels({'ICA007': 'MotorICA'})
del cleaned_epochs7, ica7

cleaned_epochs10 = mne.read_epochs('Data\cleaned_data_epochs_10.fif', preload=True)
ica10 = read_ica('Data\ICAsub10infomax-ica.fif')
ICAmotor10 = ica10.get_sources(cleaned_epochs10).pick("ICA008").rename_channels({'ICA008': 'MotorICA'})
del cleaned_epochs10, ica10

cleaned_epochs13 = mne.read_epochs('Data\cleaned_data_epochs_13.fif', preload=True)
ica13 = read_ica('Data\ICAsub13infomax-ica.fif')
ICAmotor13 = ica13.get_sources(cleaned_epochs13).pick("ICA006").rename_channels({'ICA006': 'MotorICA'})
del cleaned_epochs13, ica13

#cleaned_epochs20 = mne.read_epochs('Data\cleaned_data_epochs_20.fif', preload=True)
#ica20 = read_ica('Data\ICAsub20infomax-ica.fif')
#ICAmotor20 = ica20.get_sources(cleaned_epochs20).pick("ICA002").rename_channels({'ICA002': 'MotorICA'})
#del cleaned_epochs20, ica20

cleaned_epochs21 = mne.read_epochs('Data\cleaned_data_epochs_21.fif', preload=True)
ica21 = read_ica('Data\ICAsub21infomax-ica.fif')
ICAmotor21 = ica21.get_sources(cleaned_epochs21).pick("ICA009").rename_channels({'ICA009': 'MotorICA'})
del cleaned_epochs21, ica21

cleaned_epochs24 = mne.read_epochs('Data\cleaned_data_epochs_24.fif', preload=True)
ica24 = read_ica('Data\ICAsub24infomax-ica.fif')
ICAmotor24 = ica24.get_sources(cleaned_epochs24).pick("ICA007").rename_channels({'ICA007': 'MotorICA'})
del cleaned_epochs24, ica24

cleaned_epochs28 = mne.read_epochs('Data\cleaned_data_epochs_28.fif', preload=True)
ica28 = read_ica('Data\ICAsub28infomax-ica.fif')
ICAmotor28 = ica28.get_sources(cleaned_epochs28).pick("ICA008").rename_channels({'ICA008': 'MotorICA'})
del cleaned_epochs28, ica28

cleaned_epochs31 = mne.read_epochs('Data\cleaned_data_epochs_31.fif', preload=True)
ica31 = read_ica('Data\ICAsub31infomax-ica.fif')
ICAmotor31 = ica31.get_sources(cleaned_epochs31).pick("ICA001").rename_channels({'ICA001': 'MotorICA'})
del cleaned_epochs31, ica31

cleaned_epochs33 = mne.read_epochs('Data\cleaned_data_epochs_33.fif', preload=True)
ica33 = read_ica('Data\ICAsub33infomax-ica.fif')
ICAmotor33 = ica33.get_sources(cleaned_epochs33).pick("ICA007").rename_channels({'ICA007': 'MotorICA'})
del cleaned_epochs33, ica33

cleaned_epochs34 = mne.read_epochs('Data\cleaned_data_epochs_34.fif', preload=True)
ica34 = read_ica('Data\ICAsub34infomax-ica.fif')
ICAmotor34 = ica34.get_sources(cleaned_epochs34).pick("ICA008").rename_channels({'ICA008': 'MotorICA'})
del cleaned_epochs34, ica34

cleaned_epochs39 = mne.read_epochs('Data\cleaned_data_epochs_39.fif', preload=True)
ica39 = read_ica('Data\ICAsub39infomax-ica.fif')
ICAmotor39 = ica39.get_sources(cleaned_epochs39).pick("ICA002").rename_channels({'ICA002': 'MotorICA'})
del cleaned_epochs39, ica39


#%%

AllSubMotors = mne.concatenate_epochs([ICAmotor7, ICAmotor10, ICAmotor13, ICAmotor21, 
                                       ICAmotor24, ICAmotor28, ICAmotor31, ICAmotor33, 
                                       ICAmotor34, ICAmotor39],  on_mismatch="ignore")

del ICAmotor7, ICAmotor10, ICAmotor13, ICAmotor21, ICAmotor24, ICAmotor28, ICAmotor31, ICAmotor33, ICAmotor34, ICAmotor39


#%%
# Motor conditions
motor_conditions = {
    '41': AllSubMotors['41'],
    '611': AllSubMotors['611'],
    '612': AllSubMotors['612'],
    '613': AllSubMotors['613']
}

# Define alpha and beta frequencies
freqs_alpha = np.arange(8, 13)  # 8 to 13 Hz for alpha
freqs_beta = np.arange(15, 25)  # 15 to 25 Hz for beta
n_cycles_alpha = freqs_alpha/3
n_cycles_beta = freqs_beta/3

# Initialize a dictionary to store the ERD data
erd_storage = {}

for i, (condition, data) in enumerate(motor_conditions.items()):
    # Calculate power for alpha and beta frequencies
    power_alpha = mne.time_frequency.tfr_morlet(data, freqs_alpha, n_cycles_alpha, use_fft=False, return_itc=False,
                                                decim=1, zero_mean=True, average=False, output='power', picks='all')
    power_beta = mne.time_frequency.tfr_morlet(data, freqs_beta, n_cycles_beta, use_fft=False, return_itc=False,
                                               decim=1, zero_mean=True, average=False, output='power', picks='all')
    
    # Calculate baseline
    time_alpha = (power_alpha.times >= -0.5) & (power_alpha.times <= 1.5)
    time_beta = (power_beta.times >= -0.5) & (power_beta.times <= 1.5)
    
    baseline_mean_alpha = power_alpha.data[..., time_alpha].mean(axis=-1).mean(axis=2)
    baseline_mean_beta = power_beta.data[..., time_beta].mean(axis=-1).mean(axis=2)
    
    # Calculate mean and ERD
    mean_alpha = power_alpha.data.mean(axis=2).squeeze(axis=1)
    mean_beta = power_beta.data.mean(axis=2).squeeze(axis=1)
    
    erd_alpha = ((mean_alpha[..., time_alpha] - baseline_mean_alpha) / baseline_mean_alpha)
    erd_beta = ((mean_beta[..., time_beta] - baseline_mean_beta) / baseline_mean_beta)
    
    # Averaging across samples
    erd_alpha_avg = erd_alpha.mean(axis=0)
    erd_beta_avg = erd_beta.mean(axis=0)
    
    del erd_alpha, erd_beta
    # Store results in arrays
    alpha_erd_array = np.column_stack((power_alpha.times[time_alpha], erd_alpha_avg))
    beta_erd_array = np.column_stack((power_beta.times[time_beta], erd_beta_avg))

    erd_storage[f'{condition}_alpha'] = alpha_erd_array
    erd_storage[f'{condition}_beta'] = beta_erd_array

    del power_alpha, power_beta, mean_alpha, mean_beta

# N*2 dimensional numpy array containing the ERD for all 4 conditions for alpha and beta
#%%
#del AllSubMotors

# Define the time window
start_time = -0.25
end_time = 0.75

alpha_erd_41 = erd_storage['41_alpha']
alpha_erd_41 = alpha_erd_41[(alpha_erd_41[:, 0] >= start_time) & (alpha_erd_41[:, 0] <= end_time)]

beta_erd_41 = erd_storage['41_beta']
beta_erd_41 = beta_erd_41[(beta_erd_41[:, 0] >= start_time) & (beta_erd_41[:, 0] <= end_time)]

alpha_erd_611 = erd_storage['611_alpha']
alpha_erd_611 = alpha_erd_611[(alpha_erd_611[:, 0] >= start_time) & (alpha_erd_611[:, 0] <= end_time)]

beta_erd_611 = erd_storage['611_beta']
beta_erd_611 = beta_erd_611[(beta_erd_611[:, 0] >= start_time) & (beta_erd_611[:, 0] <= end_time)]

alpha_erd_612 = erd_storage['612_alpha']
alpha_erd_612 = alpha_erd_612[(alpha_erd_612[:, 0] >= start_time) & (alpha_erd_612[:, 0] <= end_time)]

beta_erd_612 = erd_storage['612_beta']
beta_erd_612 = beta_erd_612[(beta_erd_612[:, 0] >= start_time) & (beta_erd_612[:, 0] <= end_time)]

alpha_erd_613 = erd_storage['613_alpha']
alpha_erd_613 = alpha_erd_613[(alpha_erd_613[:, 0] >= start_time) & (alpha_erd_613[:, 0] <= end_time)]

beta_erd_613 = erd_storage['613_beta']
beta_erd_613 = beta_erd_613[(beta_erd_613[:, 0] >= start_time) & (beta_erd_613[:, 0] <= end_time)]

# %%
# Create the info structure needed by MNE
info = mne.create_info(ch_names=['MotorICA'], sfreq=257, ch_types='misc')

# Create EpochsArray objects for each condition
epochs_alpha_41 = mne.EpochsArray(data=alpha_erd_41[:, np.newaxis, :], info=info)  # new axis for channel
epochs_alpha_611 = mne.EpochsArray(data=alpha_erd_611[:, np.newaxis, :], info=info)  # new axis for channel
epochs_alpha_612 = mne.EpochsArray(data=alpha_erd_612[:, np.newaxis, :], info=info)  # new axis for channel
epochs_alpha_613 = mne.EpochsArray(data=alpha_erd_613[:, np.newaxis, :], info=info)  # new axis for channel


stacked_alpha_data = [
    epochs_alpha_41.get_data().squeeze(1),  # squeeze out the singleton channel dimension
    epochs_alpha_611.get_data().squeeze(1),
    epochs_alpha_612.get_data().squeeze(1),
    epochs_alpha_613.get_data().squeeze(1)
]

#%%
# Assuming 'data' contains your statistical test results
mean = np.mean(stacked_alpha_data)
std_dev = np.std(stacked_alpha_data)
threshold = mean + 1.96 * std_dev  # for a 95% confidence level
#%%
from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test

# Run the permutation test
cluster_stats = permutation_cluster_test(
    stacked_alpha_data,
    n_permutations=5000,  # The number of permutations to compute
    tail=0,                # Use 1 for one-sided test
    threshold=None,        # Let MNE determine the optimal threshold
    n_jobs=1,              # Number of jobs for parallel processing; adjust as per your machine's capabilities
    out_type="mask"       # Buffer size for memory management
)

T_obs, clusters, cluster_p_values, H0 = cluster_stats

# %%
