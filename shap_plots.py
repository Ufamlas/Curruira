import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable

def shape_spectrum_interpolation(S, fs=22050):
    shape_spectrum = np.sum(S, axis=0)
    old_indices = np.arange(0,len(shape_spectrum))
    new_length = 2*fs
    new_indices = np.linspace(0,len(shape_spectrum)-1,new_length)
    spl = UnivariateSpline(old_indices,shape_spectrum,k=3,s=0)
    return spl(new_indices)

def shape_time_interpolation(S, fs=22050):
    shape_time = np.sum(S, axis=1)
    old_indices = np.arange(0,len(shape_time))
    new_length = 2*fs
    new_indices = np.linspace(0,len(shape_time)-1,new_length)
    spl = UnivariateSpline(old_indices,shape_time,k=3,s=0)
    return spl(new_indices)

def shape_spectrum_plot(i, shap_values, labels, fs=22050):
    fig, axs = plt.subplots(1, 1, figsize=(4,3))

    D = shap_values[labels[i]][i,:,:,0]
    shape_spectrum = shape_spectrum_interpolation(D, fs=fs)

    points = np.array([np.linspace(0, fs/2, num=len(shape_spectrum)), shape_spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_spectrum)), np.max(np.abs(shape_spectrum)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(5*shape_spectrum)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    axs.set_xlim(0, fs/2)
    axs.set_ylim(shape_spectrum.min()-0.001, shape_spectrum.max()+0.001)
    axs.set_xlabel('freqs [Hz]')
    axs.set_ylabel('shap values')
    axs.spines['left'].set_position(('data', 0.0))
    axs.spines['bottom'].set_position(('data', 0.0))
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    # axs.set_title(title)
    plt.show()

def shape_spectrogram_plot(i, spectrograms, shap_values, fs, nfft, winstep, winlen):
    shape_spectrogram = shap_values[1][i,:,:,0].T
    spectrogram = spectrograms[i:i+1][0,:,:,0].T

    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    S_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    librosa.display.specshow(S_db, sr = fs, n_fft=nfft, hop_length=winstep, win_length=winlen,
                            x_axis='time', y_axis='linear', vmin=np.min(S_db), vmax=np.max(S_db), cmap='coolwarm')
    plt.colorbar(pad=0.01, label='dB')
    plt.ylabel('Hz')
    plt.xlabel('time [s]')
    plt.title('signal spectrogram')

    plt.subplot(2,1,2)
    librosa.display.specshow(15*shape_spectrogram, sr = fs, n_fft=nfft, hop_length=winstep, win_length=winlen,
                            x_axis='time', y_axis='linear', vmin=-np.max(np.abs(shape_spectrogram)), vmax=np.max(np.abs(shape_spectrogram)), cmap='coolwarm')
    # plt.pcolormesh(10*shape_spectrogram,  vmin=-np.max(np.abs(shape_spectrogram)), vmax=np.max(np.abs(shape_spectrogram)), cmap='coolwarm')
    plt.title('shape spectrogram')
    plt.colorbar(pad=0.01)
    plt.xlabel('time [s]')
    plt.ylabel('Hz')
    plt.show()

def shap_signal_spectrum(i, times, wav_records, shap_values, labels, fs):
    fig, axs = plt.subplots(1, 2, figsize=(15,2))
    axs[0].plot(times, wav_records[i])
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('audio signal')
    axs[0].set_xlim([0, int(len(times)/fs)])
    axs[0].set_xlabel('time [s]')

    D = shap_values[labels[i]][i,:,:,0]
    shape_spectrum = shape_spectrum_interpolation(D, fs=fs)

    points = np.array([np.linspace(0, fs/2, num=len(shape_spectrum)), shape_spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_spectrum)), np.max(np.abs(shape_spectrum)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(5*shape_spectrum)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    axs[1].set_xlim(0, fs/2)
    axs[1].set_ylim(shape_spectrum.min()-0.001, shape_spectrum.max()+0.001)
    axs[1].set_xlabel('freqs [Hz]')
    axs[1].set_ylabel('shap values')
    axs[1].spines['left'].set_position(('data', 0.0))
    axs[1].spines['bottom'].set_position(('data', 0.0))
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    axs[1].set_title('shap spectrum')
    plt.show()

def shap_time_plot(i, times, wav_records, shap_values, labels, fs):
    D = shap_values[labels[i]][i,:,:,0]
    shape_time = shape_time_interpolation(D, fs=fs)

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([times, wav_records[i]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1, figsize=(12,2))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(3*shape_time)
    lc.set_linewidth(1)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs, label='shap values', pad=0.01)
    axs.set_xlim(0, int(len(times)/fs))
    axs.set_ylim(wav_records[i].min(), wav_records[i].max())
    axs.set_xlabel('time [s]')
    axs.set_ylabel('amplitude')
    plt.show()

def shap_signal_envelope_plot(i, times, wav_records, shap_values, labels, fs):
    D = shap_values[labels[i]][i,:,:,0]
    shape_time = shape_time_interpolation(D, fs=fs)

    analytic_signal = hilbert(wav_records[i])
    amplitude_envelope = np.abs(analytic_signal)

    points = np.array([times, amplitude_envelope]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1, figsize=(12,2))
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(3*shape_time)
    lc.set_linewidth(1.3)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs, label='shap values', pad=0.01)
    axs.set_xlim(0, int(len(times)/fs))
    # axs.set_ylim(0,8)
    axs.set_xlabel('time [s]')
    axs.set_ylabel('envelope')
    plt.show()

def shap_instantaneous_frequency_plot(i, times, wav_records, shap_values, labels, fs):
    D = shap_values[labels[i]][i,:,:,0]
    shape_time = shape_time_interpolation(D, fs=fs)

    analytic_signal = hilbert(wav_records[i])
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
    points = np.array([times[:-1], instantaneous_frequency]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1, figsize=(12,2))
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(3*shape_time)
    lc.set_linewidth(1.3)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs, label='shap values', pad=0.01)
    axs.set_xlim(0, 0.6)
    axs.set_ylim(instantaneous_frequency.min(),instantaneous_frequency.max())
    axs.set_xlabel('time [s]')
    axs.set_ylabel('instantaneous freq.')
    axs.ticklabel_format(axis='y', style='sci', scilimits=(-3,4))
    plt.show()

def shap_pitch_plot(i, wav_records, shap_values, labels, fs):
    D = shap_values[labels[i]][i,:,:,0]
    shape_time = np.sum(D, axis=1)

    pitch = librosa.yin(wav_records[i], fmin=2500, fmax=fs/2, sr=fs,
                        frame_length=512, win_length=256, hop_length=64)[4:-4]
    
    time = np.round(np.linspace(0,2,8),2)
    points = np.array([range(len(pitch)), pitch]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
 
    fig, axs = plt.subplots(1, 1, figsize=(12,2))
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(5*shape_time)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs, label='shap values', pad=0.01)
    axs.set_xlim(0, len(pitch))
    axs.set_ylim(pitch.min(),pitch.max())
    axs.set_xlabel('time [s]')
    axs.set_ylabel('pitch')
    axs.set_xticklabels(time)
    plt.show()

def shap_global_frequency_plot_vertical(freqs, shap_values, labels, fmin = 2900, fmax = 8000):
    impacted = shap_values[1][labels == 1,:,:,0]
    impacted = np.sum(impacted, axis=-2)
    impacted_avg = impacted.mean(axis=0)
    impacted_std = impacted.std(axis=0)

    non_impacted = shap_values[1][labels == 0,:,:,0]
    non_impacted = np.sum(non_impacted, axis=-2)
    non_impacted_avg = non_impacted.mean(axis=0)
    non_impacted_std = non_impacted.std(axis=0)

    plt.figure(figsize=(5,6.5))
    plt.barh(freqs, impacted_avg, yerr=impacted_std, color='r',  height=28, label="impacted",error_kw=dict(lw=1, capsize=0, capthick=1))
    plt.barh(freqs, non_impacted_avg, yerr=non_impacted_std, color='b',  height=28, label="non-impacted",error_kw=dict(lw=1, capsize=0, capthick=1))
    plt.ylim([fmin, fmax])
    plt.vlines(0, fmin, fmax, color='k', linestyles ="solid")
    plt.xlabel('shape values')
    plt.ylabel('Hz')
    plt.legend()
    plt.show()

def shap_global_frequency_plot_lines(freqs, shap_values, labels, fmin = 2900, fmax = 8000):
    impacted = shap_values[1][labels == 1,:,:,0]
    impacted = np.sum(impacted, axis=-2)
    impacted_avg = impacted.mean(axis=0)
    impacted_std = impacted.std(axis=0)

    non_impacted = shap_values[1][labels == 0,:,:,0]
    non_impacted = np.sum(non_impacted, axis=-2)
    non_impacted_avg = non_impacted.mean(axis=0)
    non_impacted_std = non_impacted.std(axis=0)

    plt.figure(figsize=(6.5,5))
    plt.plot(freqs, impacted_avg.T,'r-',  label="impacted")
    plt.plot(freqs, non_impacted_avg.T,'b-', label="non-impacted")
    plt.ylim([min(impacted_avg.min(), non_impacted_avg.min()),max(impacted_avg.max(), non_impacted_avg.max())])
    plt.xlim([fmin, fmax])
    #plt.vlines(0, fmin, fmax, color='k', linestyles ="solid")
    plt.ylabel('shape values')
    plt.xlabel('Hz')
    plt.legend()
    plt.show()

def shap_global_frequency_plot_lines_error(freqs, shap_values, labels, fmin = 2900, fmax = 8000):
    impacted = shap_values[1][labels == 1,:,:,0]
    impacted = np.sum(impacted, axis=-2)
    impacted_avg = impacted.mean(axis=0)
    impacted_std = impacted.std(axis=0)

    non_impacted = shap_values[1][labels == 0,:,:,0]
    non_impacted = np.sum(non_impacted, axis=-2)
    non_impacted_avg = non_impacted.mean(axis=0)
    non_impacted_std = non_impacted.std(axis=0)

    plt.figure(figsize=(6.5,5))
    plt.plot(freqs, impacted_avg.T,'r-',  label="impacted")
    plt.plot(freqs, impacted_avg.T+impacted_std.T,'r:')
    plt.plot(freqs, impacted_avg.T-impacted_std.T,'r:')
    plt.plot(freqs, non_impacted_avg.T,'b-', label="non-impacted")
    plt.plot(freqs, non_impacted_avg.T+non_impacted_std.T,'b:')
    plt.plot(freqs, non_impacted_avg.T-non_impacted_std.T,'b:')
    plt.ylim([min(impacted_avg.min(), non_impacted_avg.min())-max(non_impacted_std.max(),impacted_std.max()),\
              max(impacted_avg.max(), non_impacted_avg.max())+max(non_impacted_std.max(),impacted_std.max())])
    plt.xlim([fmin, fmax])
    #plt.vlines(0, fmin, fmax, color='k', linestyles ="solid")
    plt.ylabel('shape values')
    plt.xlabel('Hz')
    plt.legend()
    plt.show()

def shap_global_frequency_plot_lines_abs(freqs, shap_values, labels, fmin = 2900, fmax = 8000):
    impacted = shap_values[1][labels == 1,:,:,0]
    impacted = np.sum(impacted, axis=-2)
    impacted_avg = abs(impacted.mean(axis=0))
    impacted_std = impacted.std(axis=0)

    non_impacted = shap_values[1][labels == 0,:,:,0]
    non_impacted = np.sum(non_impacted, axis=-2)
    non_impacted_avg = abs( non_impacted.mean(axis=0))
    non_impacted_std = non_impacted.std(axis=0)

    plt.figure(figsize=(6.5,5))
    plt.plot(freqs, impacted_avg.T,'r-',  label="impacted")
    plt.plot(freqs, non_impacted_avg.T,'b-', label="non-impacted")
    plt.ylim([min(impacted_avg.min(), non_impacted_avg.min()),max(impacted_avg.max(), non_impacted_avg.max())])
    plt.xlim([fmin, fmax])
    #plt.vlines(0, fmin, fmax, color='k', linestyles ="solid")
    plt.ylabel('absolute shape values')
    plt.xlabel('Hz')
    plt.legend()
    plt.show()
def shap_global_frequency_plot_lines_abs_norm(freqs, shap_values, labels, fmin = 2900, fmax = 8000):
    impacted = shap_values[1][labels == 1,:,:,0]
    impacted = np.sum(impacted, axis=-2)
    impacted_avg = abs(impacted.mean(axis=0))
    impacted_avg /= impacted_avg.max()
    impacted_std = impacted.std(axis=0)

    non_impacted = shap_values[1][labels == 0,:,:,0]
    non_impacted = np.sum(non_impacted, axis=-2)
    non_impacted_avg = abs( non_impacted.mean(axis=0))
    non_impacted_avg /= non_impacted_avg.max()
    non_impacted_std = non_impacted.std(axis=0)

    plt.figure(figsize=(6.5,5))
    plt.plot(freqs, impacted_avg.T,'r-',  label="impacted")
    plt.plot(freqs, non_impacted_avg.T,'b-', label="non-impacted")
    plt.ylim([min(impacted_avg.min(), non_impacted_avg.min()),max(impacted_avg.max(), non_impacted_avg.max())])
    plt.xlim([fmin, fmax])
    #plt.vlines(0, fmin, fmax, color='k', linestyles ="solid")
    plt.ylabel('absolute shape values')
    plt.xlabel('Hz')
    plt.legend()
    plt.show()
def shap_global_frequency_plot(freqs, shap_values, fmin = 2800, fmax = 8000):
    i = 1
    impacted = shap_values[i][:,:,:,0]
    impacted = np.sum(impacted, axis=-2)
    avg = impacted.mean(axis=0)

    plt.figure(figsize=(15,4))

    error = impacted.var(axis=0)
    error[np.where(avg < 0)] = 0
    plt.bar(freqs, np.maximum(0, avg), yerr=error, color='r', width=28, label="impacted",error_kw=dict(lw=1, capsize=0, capthick=1))

    error = impacted.var(axis=0)
    error[np.where(avg > 0)] = 0
    plt.bar(freqs, np.minimum(0, avg), yerr=error, color='b', width=28, label="non-impacted",error_kw=dict(lw=1, capsize=0, capthick=1))
    plt.xlim([fmin, fmax])
    plt.hlines(0, fmin, fmax, color='k', linestyles ="solid")
    plt.ylabel('shape values')
    plt.xlabel('Hz')
    plt.legend()
    plt.show()

def shap_global_time_plot(i, times, shap_values, fs=22050):
    impacted = shap_values[i][:,:,:,0]
    impacted = np.sum(impacted, axis=-1)
    impacted = impacted.mean(axis=0)

    old_indices = np.arange(0,len(impacted))
    new_length = 2*fs
    new_indices = np.linspace(0,len(impacted)-1,new_length)
    spl = UnivariateSpline(old_indices,impacted,k=3,s=0)
    shape_time = spl(new_indices)


    fig, axs = plt.subplots(1, 1, figsize=(12,3))
    points = np.array([times, shape_time]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(5*shape_time)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    axs.set_xlim(0, 2)
    axs.set_ylim(-0.0002, 0.0002)
    axs.set_xlabel('time [s]')
    axs.set_ylabel('shap values')
    axs.spines['left'].set_position(('data', 0.0))
    axs.spines['bottom'].set_position(('data', 0.0))
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    axs.set_title('shap time global')
    plt.show()

def shap_time_plot2(i, times, shap_values, fs=22050):
    impacted = shap_values[i][i,:,:,0]
    
    impacted = np.sum(impacted, axis=-1)
    #impacted = impacted.mean(axis=0)

    old_indices = np.arange(0,len(impacted))
    new_length = 2*fs
    new_indices = np.linspace(0,len(impacted)-1,new_length)
    spl = UnivariateSpline(old_indices,impacted,k=3,s=0)
    shape_time = spl(new_indices)


    fig, axs = plt.subplots(1, 1, figsize=(12,3))
    points = np.array([times, shape_time]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(5*shape_time)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    axs.set_xlim(0, 2)
    axs.set_ylim(-0.010, 0.010)
    axs.set_xlabel('time')
    axs.set_ylabel('SHAP values')
    axs.spines['left'].set_position(('data', 0.0))
    axs.spines['bottom'].set_position(('data', 0.0))
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    axs.set_title('shap time ')
    plt.show()


def shap_spectogram_spectrum(i, spectrograms,  nfft, winstep, winlen, shap_values, labels, fs):
    #fig, axs = plt.subplots(1, 2, figsize=(15,2))
    fig, axs = plt.subplots(2, 1, figsize=(11,10))
    spectrogram = spectrograms[i:i+1][0,:,:,0].T

    #plt.figure(figsize=(15,2))
    #plt.subplot(1,2,1)
    S_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    #plt.figure(figsize=(5,5))
    librosa.display.specshow(S_db, sr = fs, n_fft=nfft, hop_length=winstep, win_length=winlen, ax=axs[0],
                            x_axis='time', y_axis='linear', vmin=np.min(S_db), vmax=np.max(S_db), cmap='coolwarm')
    
    fig.colorbar(ScalarMappable(Normalize(vmin=np.min(S_db), vmax=np.max(S_db), clip=False), 
                                cmap='coolwarm'), pad=0.01, label='dB', ax=axs[0])
    axs[0].set_ylabel('Hz')
    axs[0].set_xlabel('time [s]')
    axs[0].set_title('signal spectrogram')

    D = shap_values[labels[i]][i,:,:,0]
    shape_spectrum = shape_spectrum_interpolation(D, fs=fs)

    points = np.array([np.linspace(0, fs/2, num=len(shape_spectrum)), shape_spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_spectrum)), np.max(np.abs(shape_spectrum)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(5*shape_spectrum)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    axs[1].set_xlim(0, fs/2)
    #axs[1].set_ylim(shape_spectrum.min()-0.001, shape_spectrum.max()+0.001)
    axs[1].set_ylim(shape_spectrum.min()-0.001, 0.016)
    axs[1].set_xlabel('freqs [Hz]')
    axs[1].set_ylabel('shap values')
    axs[1].spines['left'].set_position(('data', 0.0))
    axs[1].spines['bottom'].set_position(('data', 0.0))
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    axs[1].set_title('shap spectrum')
    for item in ([axs[1].title, axs[1].xaxis.label, axs[1].yaxis.label] +
             axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        item.set_fontsize(13)
    plt.show()

def shap_time_plot3(i, times, shap_values, labels, spectrograms,  nfft, winstep, winlen,fs):
    #fig, axs = plt.subplots(1, 2, figsize=(15,2))
    fig, axs = plt.subplots(2, 1, figsize=(11,10))
    spectrogram = spectrograms[i:i+1][0,:,:,0].T

    #plt.figure(figsize=(15,2))
    #plt.subplot(1,2,1)
    S_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    #plt.figure(figsize=(5,5))
    librosa.display.specshow(S_db, sr = fs, n_fft=nfft, hop_length=winstep, win_length=winlen, ax=axs[0],
                            x_axis='time', y_axis='linear', vmin=np.min(S_db), vmax=np.max(S_db), cmap='coolwarm')
    
    fig.colorbar(ScalarMappable(Normalize(vmin=np.min(S_db), vmax=np.max(S_db), clip=False), 
                                cmap='coolwarm'), pad=0.01, label='dB', ax=axs[0])
    axs[0].set_ylabel('Hz')
    axs[0].set_xlabel('time [s]')
    axs[0].set_title('signal spectrogram')

    impacted = shap_values[labels[i]][i,:,:,0]
    
    impacted = np.sum(impacted, axis=-1)
    #impacted = impacted.mean(axis=0)

    old_indices = np.arange(0,len(impacted))
    new_length = 2*fs
    new_indices = np.linspace(0,len(impacted)-1,new_length)
    spl = UnivariateSpline(old_indices,impacted,k=3,s=0)
    shape_time = spl(new_indices)

    points = np.array([times, shape_time]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_time)), np.max(np.abs(shape_time)))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(5*shape_time)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    axs[1].set_xlim(0, 2)
    axs[1].set_ylim(-0.010, 0.010)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('SHAP values')
    axs[1].spines['left'].set_position(('data', 0.0))
    axs[1].spines['bottom'].set_position(('data', 0.0))
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    axs[1].set_title('shap time ')
    axs[1].set_xticks([0,0.5,1,1.5,2])
    for item in ([axs[1].title, axs[1].xaxis.label, axs[1].yaxis.label] +
             axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        item.set_fontsize(13)
    plt.show()

def shap_spectrum_superposition(i, j, shap_values, labels, fs):
    #fig, axs = plt.subplots(1, 2, figsize=(15,2))
    fig, axs = plt.subplots(1, 1, figsize=(11,10))

    D = shap_values[labels[i]][i,:,:,0]
    shape_spectrum = shape_spectrum_interpolation(D, fs=fs)

    points = np.array([np.linspace(0, fs/2, num=len(shape_spectrum)), shape_spectrum]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    D2 = shap_values[labels[j]][j,:,:,0]
    shape_spectrum2 = shape_spectrum_interpolation(D2, fs=fs)

    points2 = np.array([np.linspace(0, fs/2, num=len(shape_spectrum2)), shape_spectrum2]).T.reshape(-1, 1, 2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.max(np.abs(shape_spectrum)), np.max(np.abs(shape_spectrum)))
    lc = LineCollection(segments, cmap='bwr', norm=norm, label="Impacted")
    # Set the values used for colormapping
    lc.set_array(5*shape_spectrum)
    lc.set_linewidth(2)
    lc2 = LineCollection(segments2, cmap='PiYG', norm=norm, label="Non-impacted")
    # Set the values used for colormapping
    lc2.set_array(5*shape_spectrum2)
    lc2.set_linewidth(2)
    line = axs.add_collection(lc)
    line = axs.add_collection(lc2)
    axs.set_xlim(0, fs/2)
    #axs.set_ylim(shape_spectrum.min()-0.001, shape_spectrum.max()+0.001)
    axs.set_ylim(segments2.min()-0.001, 0.016)
    axs.set_xlabel('freqs [Hz]')
    axs.set_ylabel('shap values')
    axs.spines['left'].set_position(('data', 0.0))
    axs.spines['bottom'].set_position(('data', 0.0))
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    axs.set_title('shap spectrum superposition')
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
             axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(13)
    plt.legend()
    plt.show()

