#%%
from mne.preprocessing import read_ica
import mne
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import pandas

#Import data
cleaned_epochs = mne.read_epochs('cleaned_data_epochs.fif', preload=True)
ica = read_ica('fitted_ica-ica.fif')

#%%
ica.plot_components()
ica.plot_sources(cleaned_epochs)

#%% Define motor components for each condition
all_sources = ica.get_sources(cleaned_epochs)

motor941 = all_sources.pick("ICA009")['41']
#motor9611 = all_sources.pick(9)['611']
#motor9612 = all_sources.pick(9)['612']
#motor9613 = all_sources.pick(9)['613']

#%% Define alpha and beta frequencys 
freqs_alpha = np.arange(8, 14)  # 8 to 13 Hz for alpha
freqs_beta = np.arange(15, 26)  # 13 to 30 Hz for beta
n_cycles_alpha = 2 + 0.125 * freqs_alpha
n_cycles_beta = 2 + 0.125 * freqs_beta

# Compute TFR for alpha and beta bands
power_alpha = mne.time_frequency.tfr_morlet(motor941, freqs_alpha, n_cycles_alpha, use_fft=False, return_itc=False,
                                            decim=1, n_jobs=None, zero_mean=True, average=False, output='power', picks='all')
power_beta = mne.time_frequency.tfr_morlet(motor941, freqs_beta, n_cycles_beta, use_fft=False, return_itc=False,
                                           decim=1, n_jobs=None, zero_mean=True, average=False, output='power', picks='all')

#%% Calculate baseline
time_alpha = (power_alpha.times >= -0.5) & (power_alpha.times <= 1.5)
time_beta = (power_beta.times >= -0.5) & (power_beta.times <= 1.5)
baseline_mean_alpha = power_alpha.data[..., time_alpha].mean(axis=-1)
baseline_mean_beta = power_beta.data[..., time_beta].mean(axis=-1)

# Adding a new axis to baseline_mean_alpha to match the dimensions
baseline_mean_alpha_expanded = baseline_mean_alpha[..., np.newaxis]
baseline_mean_beta_expanded = baseline_mean_beta[..., np.newaxis]

# Now, baseline_mean_alpha_expanded has the shape (525, 1, 6, 1)
erd_alpha = (power_alpha.data[..., time_alpha] - baseline_mean_alpha_expanded) / baseline_mean_alpha_expanded
erd_beta = (power_beta.data[..., time_beta] - baseline_mean_beta_expanded) / baseline_mean_beta_expanded

erd_alpha_avg = erd_alpha.mean(axis=(0, 1, 2))
erd_beta_avg = erd_beta.mean(axis=(0, 1, 2))
#%% plot
plt.figure(figsize=(10, 6))
plt.plot(power_alpha.times[time_alpha], erd_alpha_avg, label='Alpha ERD')
plt.plot(power_beta.times[time_beta], erd_beta_avg, label='beta ERD')
plt.xlabel('Time (s)')
plt.ylabel('ERD')
plt.title('Alpha ERD Averaged Across Samples, Channels, and Frequencies (-0.5 to 1.5s)')
plt.axhline(0, color='black', lw=1, ls='--')  
plt.axvline(0, color='red', lw=1, ls='--')  
plt.legend()
plt.show()
 # %%
