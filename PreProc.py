#%%
import mne
import numpy as np
from autoreject import AutoReject
import matplotlib.pyplot as plt
from mne.preprocessing import ICA


#%% ALL PREPROCESSING FUNCTIONS
def MappingAndFiltering(scrambled_eeg):


    biosemi_to_10_20_mapping = {'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
    'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
    'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
    'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4',
    'B9': 'F6', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz',
    'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2',
    'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'}


    scrambled_eeg.rename_channels(biosemi_to_10_20_mapping)


    #set Montage
    scrambled_eeg.set_montage('standard_1020')
    print("Updated Channel Names:", scrambled_eeg.ch_names)

    SEFiltered = scrambled_eeg.filter(l_freq=1, h_freq=40, picks='eeg', verbose=True)
    return SEFiltered


def Epoching(SEFiltered):

    """
    Epochs the Data and and renaming by adaptivity 
    """

    events = mne.find_events(SEFiltered, stim_channel='Status', shortest_event=1)
    #print("Events:", events)


    events_new = []

    for row in events:
        if row[2] in [10,20,41,30,61]:
            events_new.append(row)



    # Iterate through the third column
    for i in range(len(events_new)):
        # Check if the value in the third column is 10
        if events_new[i][2] == 10:
            # Check if the next value is 61
            while i + 1 < len(events_new) and events_new[i + 1][2] == 61:
                # Change the value to 611
                events_new[i+1][2] = 611
                i += 1

    for i in range(len(events_new)):
        # Check if the value in the third column is 10
        if events_new[i][2] == 20:
            # Check if the next value is 61
            while i + 1 < len(events_new) and events_new[i + 1][2] == 61:
                # Change the value to 611
                events_new[i+1][2] = 612
                i += 1

    for i in range(len(events_new)):
        # Check if the value in the third column is 10
        if events_new[i][2] == 30:
            # Check if the next value is 61
            while i + 1 < len(events_new) and events_new[i + 1][2] == 61:
                # Change the value to 611
                events_new[i+1][2] = 613
                i += 1

    # see if there are any 61s left
    count611 = 0
    for i in range(len(events_new)):
        if events_new[i][2] == 611:
            count611 +=1
    print("Number of 611s:", count611)


    events_renamed = np.array(events_new)


    trigger_mapping = {
        '41': 41,  # Taps when not interacting
        '611': 611,  
        '612': 612,  # Taps when interacting with non-adaptive VP
        '613': 613  # Taps when interacting with moderately adaptive VP
    }

    tmin, tmax = -2, 3  # Adjust as needed
    epochs = mne.Epochs(SEFiltered, events=events_renamed, event_id=trigger_mapping, tmin=tmin, tmax=tmax, baseline=None, preload=True)

    #plot epochs
    #epochs.plot(events=events_renamed, n_epochs=3, show=True)
    return epochs


def Downsample(epochs):

    #Resampling to 256Hz
    epochs.resample(sfreq=256)
    print("done downsampling")
    return epochs


def PlotChannels(SEFiltered):

    #plot all the channels
    Channelplot1 = SEFiltered.copy().pick_channels(['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1'])
    Channelplot2 = SEFiltered.copy().pick_channels(['P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4'])
    Channelplot3 = SEFiltered.copy().pick_channels(['F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8'])
    Channelplot4 = SEFiltered.copy().pick_channels(['P10', 'PO8', 'PO4', 'O2'])

    Channelplot1.plot()
    Channelplot2.plot()
    Channelplot3.plot()
    Channelplot4.plot()

    return None


def Autoreject(epochs):

    """
    Use the autoreject package to remove the very bad epochs without interpolating the rest
    """

    #Use Autoreject to find bad epochs
    ar = AutoReject()
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)
    #epochs_clean = ar.transform(epochs)

    ##Plot the rejected epochs
    #Drop bad epochs manually using the reject_log
    reject_log.plot_epochs(epochs)
    epochs.drop(reject_log.bad_epochs)

    #n_epochs_before = len(epochs)

    # Assuming 'epochs_clean' is the result after applying AutoReject
    #n_epochs_after = len(epochs_clean)

    # Calculate the number of dropped epochs
    #n_epochs_dropped = n_epochs_before - n_epochs_after

    #print(f"Number of epochs before AutoReject: {n_epochs_before}")
    #print(f"Number of epochs after AutoReject: {n_epochs_after}")
    #print(f"Number of epochs dropped: {n_epochs_dropped}")

    return epochs


def CAvgRef(epochs):

    """
    Minus the common average reference
    """
    AVGSEF = epochs.set_eeg_reference('average', projection=True)
    AVGSEF.apply_proj()

    AVGSEF.plot(title = "common AVG plot")

    """
    # Average epoch for each condition
    epoch41 = AVGSEF["41"].average()
    epoch41.plot_joint(title = "41 joint plot")

    epoch611 = AVGSEF["611"].average()
    epoch611.plot_joint(title = "611 joint plot")

    epoch612 = AVGSEF["612"].average()
    epoch612.plot_joint(title = "612 joint plot")

    epoch613 = AVGSEF["613"].average()
    epoch613.plot_joint(title = "613 joint plot")
    """

    return AVGSEF


def PerformICA(AVGSEF):
    ica = ICA(n_components=32, method = 'infomax', max_iter="auto", random_state=97)
    ica.fit(AVGSEF)

    explained_var_ratio = ica.get_explained_variance_ratio(AVGSEF)
    print("Fraction of variance explained by all components:", explained_var_ratio)


    # Select an epoch to plot
    ica.plot_sources(AVGSEF, show_scrollbars=False)
    ica.plot_components()

    return ica


def PlotICA(ica, AVGSEF, ICAPicks):

    ica.plot_properties(AVGSEF, picks=ICAPicks)
    return None



#%% Load BDF file
bdf_file_path = 'Data\subj_40_230223_task.bdf'

Raw_eeg = mne.io.read_raw_bdf(bdf_file_path, include=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 
                                                  'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
                                                  'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 
                                                  'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'Status'], preload=True)
print("done reading")


#%%Get filtered data
FilteredData = MappingAndFiltering(Raw_eeg)
del Raw_eeg

#%% Epoch the data
InitialEpochs = Epoching(FilteredData)

#%% Downsample
InitialEpochs = Downsample(InitialEpochs)

#%% Inspect channels for bad channels
InitialEpochs.plot_psd(fmax=50)
plt.show()


#%%
#InitialEpochs.info['bads'] = ['PO3']  # Temporarily set to only include target bad channel
#InitialEpochs.interpolate_bads(reset_bads=True)
#InitialEpochs.plot_psd(fmax=50)
#plt.show()
#PlotChannels(InitialEpochs)
#del FilteredData


#%% Reject trails using autoreject
CleanEpochs = Autoreject(InitialEpochs)
del InitialEpochs

#%% Common average reference and perform ICA
CARepochs = CAvgRef(CleanEpochs)
del CleanEpochs
CARepochs.info

FittetICA = PerformICA(CARepochs)

#%%List of the motor components to inspect
ICAPicks = [2,6]
PlotICA(FittetICA, CARepochs, ICAPicks)



# %% Save the cleaned data and ICA
CARepochs.save('cleaned_data_epochs_39.fif',overwrite=True)
FittetICA.save('ICAsub39infomax-ica.fif')




# %%
