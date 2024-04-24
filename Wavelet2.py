#%%
from mne.preprocessing import read_ica
import mne
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import pandas
from mne.preprocessing import ICA
from mne.time_frequency import (
    AverageTFR,
    tfr_array_morlet,
    tfr_morlet,
    tfr_multitaper,
    tfr_stockwell,
)

cleaned_epochs = mne.read_epochs('Data\cleaned_data_epochs_39.fif', preload=True)

ica = read_ica('Data\ICAsub39infomax-ica.fif')

#ica20 = read_ica('Data\ICAsub13infomax-ica.fif')
ica.plot_components()

#%%
ICAmotor = ica.get_sources(cleaned_epochs).pick("ICA002")

#%%
# Motor conditions
motor_conditions = {
    '41': ICAmotor['41'],
    '611': ICAmotor['611'],
    '612': ICAmotor['612'],
    '613': ICAmotor['613']
}

# Define alpha and beta frequencies
freqs_alpha = np.arange(8, 13)  # 8 to 13 Hz for alpha
freqs_beta = np.arange(15, 25)  # 15 to 25 Hz for beta
n_cycles_alpha = freqs_alpha / 3
n_cycles_beta = freqs_beta / 3

# Setup plot
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
axs = axs.flatten()  # Flatten the 2x2 matrix to easily iterate over it

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
    
    erd_alpha = ((mean_alpha[..., time_alpha] - baseline_mean_alpha) / baseline_mean_alpha)*100
    erd_beta = ((mean_beta[..., time_beta] - baseline_mean_beta) / baseline_mean_beta)*100
    
    # Averaging across samples
    erd_alpha_avg = erd_alpha.mean(axis=0)
    erd_beta_avg = erd_beta.mean(axis=0)
    
    # Plot
    ax = axs[i]
    ax.plot(power_alpha.times[time_alpha], erd_alpha_avg, label='Alpha ERD')
    ax.plot(power_beta.times[time_beta], erd_beta_avg, label='Beta ERD')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ERD')
    ax.set_title(f'{condition} Alpha and Beta ERD (-0.5 to 1.5s)')
    ax.axhline(0, color='black', lw=1, ls='--')  
    ax.axvline(0, color='red', lw=1, ls='--')  
    ax.legend()

plt.tight_layout()
plt.show()
# %%
