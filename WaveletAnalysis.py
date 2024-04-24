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
                                       ICAmotor24, ICAmotor28, ICAmotor31, ICAmotor33, ICAmotor34],  on_mismatch="ignore")

del ICAmotor7
del ICAmotor10
del ICAmotor13
del ICAmotor21
del ICAmotor24
del ICAmotor28
del ICAmotor31
del ICAmotor33
del ICAmotor34

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


# Setup plot
#fig, axs = plt.subplots(2, 2, figsize=(20, 12))
#axs = axs.flatten()  # Flatten the 2x2 matrix to easily iterate over it
# Setup plot for combined alpha and beta
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
alpha_ax, beta_ax = axs 


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
    erd_alpha_sem = erd_alpha.std(axis=0) / np.sqrt(erd_alpha.shape[0])
    erd_beta_sem = erd_beta.std(axis=0) / np.sqrt(erd_beta.shape[0])

    # Plot Alpha and Beta ERD with error shading
    alpha_times = power_alpha.times[time_alpha]
    beta_times = power_beta.times[time_beta]

    alpha_ax.plot(alpha_times, erd_alpha_avg, label=f'Condition {condition} Alpha')
    alpha_ax.fill_between(alpha_times, erd_alpha_avg - erd_alpha_sem, erd_alpha_avg + erd_alpha_sem, alpha=0.2)

    beta_ax.plot(beta_times, erd_beta_avg, label=f'Condition {condition} Beta')
    beta_ax.fill_between(beta_times, erd_beta_avg - erd_beta_sem, erd_beta_avg + erd_beta_sem, alpha=0.2)

# Enhance the plots with labels and legends
alpha_ax.set_title('Alpha ERD (-0.5 to 1.5s)')
alpha_ax.set_xlabel('Time (s)')
alpha_ax.set_ylabel('ERD (%)')
alpha_ax.axhline(0, color='black', lw=1, ls='--')
alpha_ax.axvline(0, color='red', lw=1, ls='--')
alpha_ax.legend()

beta_ax.set_title('Beta ERD (-0.5 to 1.5s)')
beta_ax.set_xlabel('Time (s)')
beta_ax.set_ylabel('ERD (%)')
beta_ax.axhline(0, color='black', lw=1, ls='--')
beta_ax.axvline(0, color='red', lw=1, ls='--')
beta_ax.legend()

plt.tight_layout()
plt.show()

# %%
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
    erd_alpha_sem = erd_alpha.std(axis=0) / np.sqrt(erd_alpha.shape[0])
    erd_beta_sem = erd_beta.std(axis=0) / np.sqrt(erd_beta.shape[0])

    # Plot with error shading
    ax = axs[i]
    alpha_times = power_alpha.times[time_alpha]
    beta_times = power_beta.times[time_beta]

    ax.plot(alpha_times, erd_alpha_avg, label='Alpha ERD')
    ax.fill_between(alpha_times, erd_alpha_avg - erd_alpha_sem, erd_alpha_avg + erd_alpha_sem, alpha=0.2)
    ax.plot(beta_times, erd_beta_avg, label='Beta ERD')
    ax.fill_between(beta_times, erd_beta_avg - erd_beta_sem, erd_beta_avg + erd_beta_sem, alpha=0.2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ERD (%)')
    ax.set_title(f'{condition} Alpha and Beta ERD (-0.5 to 1.5s)')
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.axvline(0, color='red', lw=1, ls='--')
    ax.legend()

plt.tight_layout()
plt.show()
# %%
