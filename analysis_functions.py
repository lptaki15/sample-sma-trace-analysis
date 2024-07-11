"""
Created on Tues Jan 23 2024
@author: ltakiguchi

functions:
load_csv_data(csv_path, pixel size)
plot_1d_raw_data(data_crop, fsample, save_folder, csv_number, exp_name, save_figs)
plot_kde(t_data_crop, save_folder, csv_number, exp_name, fsample, save_figs)
plot_2d_scatter(t_data_crop, save_folder, csv_number, exp_name, fsample, save_figs))
plot_psd(t_data_crop, fsample, save_folder, csv_number, exp_name, save_figs)
analyze_folder(csv_filepath, pixel_size, fsample, save_folder, save_figs)

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tweezepy import PSD

# LOAD CSVS FROM FOLDER ###########
def load_csv_data(csv_path, pixel_size):
    data_raw = pd.read_csv(csv_path, names=['xx', 'yy', 'c0int', 'fitheight', 'noise', 'SNR'])
    data_raw = data_raw[(data_raw['xx'] != 0) & (data_raw['yy'] != 0)]
    data_raw['xpos_nm'] = data_raw['xx'] * pixel_size
    data_raw['ypos_nm'] = data_raw['yy'] * pixel_size
    data_crop = data_raw.head(-100)
    return data_crop

# PLOT 1D RAW DATA ################
def plot_1d_raw_data(data_crop, fsample, save_folder, csv_number, exp_name, save_figs):
    try:
        fig, ax = plt.subplots(figsize=(10, 8), ncols=2, nrows=2, gridspec_kw={'width_ratios': [2.75, .75], 'hspace': 0.35, 'wspace': 0})
        x_com, y_com = data_crop['xpos_nm'].mean(), data_crop['ypos_nm'].mean()
        dx, dy = data_crop['xpos_nm'] - x_com, data_crop['ypos_nm'] - y_com
        N = len(data_crop['xpos_nm'])
        time = np.arange(N) / fsample
        t_data_crop = pd.DataFrame({'xpos_nm': data_crop['xpos_nm'], 'ypos_nm': data_crop['ypos_nm'], 'xpos_nm_t': dx, 'ypos_nm_t': dy,
                                   'c0int': data_crop['c0int'], 'fitheight': data_crop['fitheight'], 'noise': data_crop['noise'], 'SNR': data_crop['SNR']})

        for i in range(2):
            ax[i, 0].plot(time, t_data_crop[f'{["xpos_nm_t", "ypos_nm_t"][i]}'], label=f'raw {["x(t)", "y(t)"][i]}')
            ax[i, 0].set_xlabel('t (s)', fontsize=16, fontweight='bold')
            ax[i, 0].set_ylabel(f'{["x(t) (nm)", "y(t) (nm)"][i]}', fontsize=16, fontweight='bold')
            ax[i, 0].legend(fontsize=8)
            hist = ax[i, 1].hist([dx, dy][i], bins=100, orientation='horizontal')
            ax[i, 1].set_xticks([])
            ax[i, 1].set_yticks([])
            std_val = np.std([t_data_crop['xpos_nm_t'], t_data_crop['ypos_nm_t']][i])
            ax[i, 1].annotate("raw stdev (nm) = {:.2f}".format(std_val), xy=(0.05, 0.98), xycoords='axes fraction', ha='left', va='top', fontsize=8)

        new_fsample = 10
        downsample_factor = int(fsample/new_fsample)
        downsampled_data = t_data_crop[['xpos_nm_t', 'ypos_nm_t']].rolling(window=downsample_factor).mean()
        downsampled_time = np.linspace(0, len(t_data_crop)/fsample, len(downsampled_data))

        for i in range(2):
            ax[i, 0].plot(downsampled_time, downsampled_data.iloc[:, i], label=f'downsampled {["x(t)", "y(t)"][i]}', color='lightblue')
            hist = ax[i, 1].hist(downsampled_data.iloc[:, i], bins=100, color='lightblue', orientation='horizontal')
            std_downsample = np.std(downsampled_data.iloc[:, i])
            ax[i, 1].annotate('new stdev (nm) = {:.2f}'.format(std_downsample), xy=(0.05, 0.94), xycoords='axes fraction', ha='left', va='top', fontsize=8)

        ax[0, 0].legend(fontsize=8)
        ax[1, 0].legend(fontsize=8)
        ax[0, 0].set_title(f'Trace: {csv_number}, fs: {fsample} Hz\n Exp Name: {exp_name}', fontsize=20, fontweight='bold')

        if save_figs:
            fig1 = f'{save_folder}{csv_number}_1d.png'
            plt.savefig(fig1)
            plt.close(fig)

        return t_data_crop

    except Exception as e:
        print(f"Error processing {csv_number}: {e}")
        return None

# PLOT 2D SCATTER ##################
def plot_2d_scatter(t_data_crop, save_folder, csv_number, exp_name, fsample, save_figs):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(t_data_crop['xpos_nm_t'], t_data_crop['ypos_nm_t'], c=np.arange(len(t_data_crop)), cmap='Blues')
        ax.set_xlabel('x(t) (nm)', fontweight='bold', fontsize=14)
        ax.set_ylabel('y(t) (nm)', fontweight='bold', fontsize=14)
        ax.set_aspect('equal')
        ax.set_title(f'Trace: {csv_number}, fs: {fsample} Hz\n Exp Name: {exp_name}', fontsize=16, fontweight='bold')

        if save_figs:
            fig1 = f'{save_folder}{csv_number}_2dscatter.png'
            plt.savefig(fig1)
            plt.close(fig)

    except Exception as e:
        print(f"Error processing 2D scatter plot for {csv_number}: {e}")

# 2D GAUSSIAN KDE PLOT ##########
def plot_kde(t_data_crop, save_folder, csv_number, exp_name, fsample, save_figs): 
    try:
        # Check if t_data_crop is None
        if t_data_crop is None:
            print(f"Skipping KDE plot for {csv_number} as t_data_crop is None.")
            return
        fig, ax = plt.subplots()
        data = t_data_crop
        x_kde = t_data_crop['xpos_nm_t']
        y_kde = t_data_crop['ypos_nm_t']
        kde = sns.kdeplot(data=data, x=x_kde, y=y_kde, cmap='Blues', fill=True, thresh=0, cbar=True)
        ax.set_xlabel('x (nm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('y (nm)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        plt.title(f'KDE of Trace: {csv_number}, fs: {fsample} Hz\n Exp Name: {exp_name}', fontweight='bold', fontsize=16) 
        if save_figs:
            fig1 = f'{save_folder}{csv_number}_kde.png'
            plt.savefig(fig1)
            plt.close(fig)
    except Exception as e:
        print(f"Error processing KDE for {csv_number}: {e}")

# PLOT PSD ########################
def plot_psd(t_data_crop, fsample, save_folder, csv_number, exp_name, save_figs):
    try:
        binsize = 10
        psdtrace_x = t_data_crop['xpos_nm_t']
        psdtrace_y = t_data_crop['ypos_nm_t']
        psdtrace_c0int = t_data_crop['c0int']
        N_psd = len(psdtrace_x)
        psdtrace_time = np.arange(N_psd)/fsample
        psd_x = PSD(psdtrace_x, fsample, bins=binsize)
        psd_y = PSD(psdtrace_y, fsample, bins=binsize)
        psd_c0int = PSD(psdtrace_c0int, fsample, bins=binsize)
        psd_x.mlefit()
        psd_y.mlefit()
        psd_c0int.mlefit()
        psd_results_x = pd.DataFrame([psd_x.results])
        psd_results_x['csv_number'] = csv_number
        psd_results_x['component'] = 'x'
        psd_results_y = pd.DataFrame([psd_y.results])
        psd_results_y['csv_number'] = csv_number
        psd_results_y['component'] = 'y'
        psd_results_c0int = pd.DataFrame([psd_c0int.results])
        psd_results_c0int['csv_number'] = csv_number
        psd_results_c0int['component'] = 'c0int'

        df_combined = pd.concat([psd_results_x, psd_results_y, psd_results_c0int], ignore_index=True)
        df_combined.to_csv(os.path.join(save_folder, exp_name+'_psdresults.txt'), sep='\t', index=False, mode='a', header=not os.path.exists(os.path.join(save_folder, exp_name+'_psdresults.txt')))

        fig, ax = psd_x.plot(data_label='Raw Data', fit_label='MLE Fit', data_color='steelblue')
        ax[1].set_xlabel('f (Hz)', fontsize=14, fontweight='bold', labelpad=0.01)
        ax[0].set_ylabel(r'PSD ($\mathbf{nm^2/{Hz}}$)', fontsize=14, fontweight='bold', labelpad=0.01)
        ax[0].set_title(f'PSD of x - {csv_number}', fontsize=18, fontweight='bold', y=1.03)
        ax[0].legend(fontsize=8)
        fig.set_size_inches(5, 5)
        plt.tight_layout()

        if save_figs:
            fig1 = f'{save_folder}{csv_number}_PSD-x.png'
            plt.savefig(fig1)
        plt.close(fig)

        fig, ax = psd_y.plot(data_label='Raw Data', fit_label='MLE Fit', data_color='steelblue')
        ax[1].set_xlabel('f (Hz)', fontsize=14, fontweight='bold', labelpad=0.01)
        ax[0].set_ylabel(r'PSD ($\mathbf{nm^2/{Hz}}$)', fontsize=14, fontweight='bold', labelpad=0.01)
        ax[0].set_title(f'PSD of y - {csv_number}', fontsize=18, fontweight='bold', y=1.03)
        ax[0].legend(fontsize=8)
        fig.set_size_inches(5, 5)
        plt.tight_layout()

        if save_figs:
            fig1 = f'{save_folder}{csv_number}_PSD-y.png'
            plt.savefig(fig1)
        plt.close(fig)

        fig, ax = psd_c0int.plot(data_label='Raw Data', fit_label='MLE Fit', data_color='steelblue')
        ax[1].set_xlabel('f (Hz)', fontsize=14, fontweight='bold', labelpad=0.01)
        ax[0].set_ylabel(r'PSD ($\mathbf{units^2/{Hz}}$)', fontsize=14, fontweight='bold', labelpad=0.01)
        ax[0].set_title(f'PSD of c0int - {csv_number}', fontsize=18, fontweight='bold', y=1.03)
        ax[0].legend(fontsize=8)
        fig.set_size_inches(5, 5)
        plt.tight_layout()

        if save_figs:
            fig1 = f'{save_folder}{csv_number}_PSD-c0int.png'
            plt.savefig(fig1)
        plt.close(fig)

    except Exception as e:
        print(f"Error processing PSD for {csv_number}: {e}")

# ANALYZE FOLDER OF CSVS ##############
def analyze_folder(csv_filepath, pixel_size, fsample, save_folder, save_figs):
    os.makedirs(save_folder, exist_ok=True)
    csv_files = [file for file in os.listdir(csv_filepath) if file.endswith('.csv')]
    for csv_file in csv_files:
        csv_path = os.path.join(csv_filepath, csv_file)
        csv_number = os.path.splitext(csv_file)[0]
        exp_name = csv_filepath.split('/')[-2]

        data_crop = load_csv_data(csv_path, pixel_size)
        t_data_crop = plot_1d_raw_data(data_crop, fsample, save_folder, csv_number, exp_name, save_figs)
        plot_kde(t_data_crop, save_folder, csv_number, exp_name, fsample, save_figs)
        plot_2d_scatter(t_data_crop, save_folder, csv_number, exp_name, fsample, save_figs)
        #plot_psd(t_data_crop, fsample, save_folder, csv_number, exp_name, save_figs)

    plt.close('all')
    print('Folder analyzed successfully')