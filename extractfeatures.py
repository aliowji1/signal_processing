import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

#converting to binary
file_path = ''
data = pd.read_csv(file_path)

columns_to_convert = ['spg_filt_upsampled_peaks', 'spg_filt_upsampled_troughs']  

# Convert the specified columns to binary
for column in columns_to_convert:
    data[column] = data[column].notna().astype(int)

data.to_csv('', index=False)

file_path = ""
df = pd.read_csv(file_path)

def get_3_SPGwaveforms(df_in, trough_indices, i):

    """

    Finds three subsequent waveforms, centered on the waveform of interest (i)
 
    Parameters:

    df (pd.DataFrame): The input DataFrame containing the signal data.

    trough_indices (pd.Index): The indices of the labelled toughs.
 
    Returns:

    tuple: A tuple containing:

        - pd.DataFrame: The DataFrame containing the three subsequent waveforms.

        - int: The start index of the central waveform.

        - int: The end index of the central waveform.
 
    """

    if i - 1 <= 0 or i + 2 >= len(trough_indices):

        print("No prior/post waveform, skipping")

        return None, None, None
 
    start_of_3_waveforms = trough_indices[i-1]

    end_of_3_waveforms = trough_indices[i + 2]
 
    start_of_central_waveform = trough_indices[i]

    end_of_central_waveform = trough_indices[i+1]

    return df_in.loc[start_of_3_waveforms:end_of_3_waveforms - 1], start_of_central_waveform, end_of_central_waveform

def calculate_sampling_rate(df_in):

    df = df_in.copy()

    if len(df) > 1:

        time_diffs = df['time'].diff().dropna()

        mean_diff = time_diffs.mean()

        sampling_rate = 1 / mean_diff

    return(sampling_rate)
 

def smoothing_filter_for_derivatives(df_in, col_name, sampling_rate, polyorder=3):

    # Avoid SettingWithCopy warning

    df = df_in.copy()
 
    # Determine appropriate window_length for the smoothing filter, based on data sampling rate

    window_length = max(3, int(sampling_rate / 10))  # Ensure the window length is at least 3. Factor of 10 is empirically determined. Smaller number increases smoothing

    if window_length % 2 == 0:  # Ensure the window length is odd

        window_length += 1

    df[col_name] = savgol_filter(df[col_name], window_length, polyorder)

    return df
 

def calculate_derivatives(df, column_name, no_derivatives=1):

    # Avoid SettingWithCopy warning

    df_in = df.copy()

    base_column_name = column_name

    for x in range(1, no_derivatives + 1):

        df_in['time_diff'] = df_in['normalized_time'].diff()  # Calculate time differences

        new_column_name = f'{column_name}_{x}_derivative'

        df_in[new_column_name] = df_in[base_column_name].diff() / df_in['time_diff']  # Calculate derivative

        df_in.drop(columns=['time_diff'], inplace=True)  # Remove the time_diff column
 
        sampling_rate = calculate_sampling_rate(df_in)

        df_in = smoothing_filter_for_derivatives(df_in, new_column_name, sampling_rate)

        base_column_name = new_column_name  # Update column name for next iteration
 
    return df_in
 
def extract_central_waveform(df, start_of_central_waveform, end_of_central_waveform):
 
    central_waveform = df.loc[start_of_central_waveform : end_of_central_waveform].copy()
 
    return central_waveform
 
#----------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------
 
# Name of the column to normalize

spg_col_name = ['spg_filt_upsampled']
 
# Identify trough indices, the start of each aveform. This assumes the trough column is full of NaNs except where there are troughs.

# If you've changed this column to be all zeros except where there are troughs, then use the second, commented line

# trough_indices = df.loc[df[f'{spg_col_name[0]}_troughs'].notnull()].index

trough_indices = df.loc[df[f'{spg_col_name[0]}_troughs'] != 0].index
print(f'these are the trough indices {trough_indices} ')
# Get waveform

three_waveforms, start_of_central_waveform, end_of_central_waveform = get_3_SPGwaveforms(df, trough_indices, 150)
 
if three_waveforms is None:

    print("No valid waveforms found, skipping normalization.")

else:

    # Copy waveform to avoid SettingWithCopyWarning

    three_waveforms = three_waveforms.copy()

    # Initialize MinMaxScaler for normalization between 0 and 1 for y-axis and time

    scaler_amplitude = MinMaxScaler(feature_range=(0, 1))

    scaler_time = MinMaxScaler(feature_range=(0, 1))
 
    # Check if the waveform DataFrame has the expected column

    if f'{spg_col_name[0]}' in three_waveforms.columns:

        # Isolate the central waveform for amplitude normalization

        central_waveform_amplitude = three_waveforms.loc[start_of_central_waveform:end_of_central_waveform, f'{spg_col_name[0]}'].values.reshape(-1, 1)

        central_waveform_time = three_waveforms.loc[start_of_central_waveform:end_of_central_waveform, 'time'].values.reshape(-1, 1)
 
        # Normalize the central waveform for amplitude

        scaler_amplitude.fit(central_waveform_amplitude)

        normalized_central_waveform_amplitude = scaler_amplitude.transform(central_waveform_amplitude)
 
        # Normalize the central waveform for time

        scaler_time.fit(central_waveform_time)

        normalized_central_waveform_time = scaler_time.transform(central_waveform_time)
 
        # Apply the same scaling to the entire three waveforms for amplitude and time

        three_waveforms[f'normalized_{spg_col_name[0]}'] = scaler_amplitude.transform(three_waveforms[[f'{spg_col_name[0]}']])

        three_waveforms['normalized_time'] = scaler_time.transform(three_waveforms[['time']])
 
        # # Plotting the normalized waveform

        # plt.plot(three_waveforms['normalized_time'], three_waveforms[f'normalized_{spg_col_name[0]}'])

        # plt.xlabel('Normalized Time')

        # plt.ylabel('Normalized SPG Amplitude')

        # plt.title('Normalized SPG Waveform')

        # plt.show()

    else:

        raise ValueError(f"Column {spg_col_name[0]} not found in waveform DataFrame.")
 
# Use calculate_derivatives function

three_waveforms = calculate_derivatives(three_waveforms, f'normalized_{spg_col_name[0]}', no_derivatives=4)
 
# Extract just the central waveform

central_waveform = extract_central_waveform(three_waveforms, start_of_central_waveform, end_of_central_waveform)
 
# Plotting the normalized waveform

fig, ax = plt.subplots(figsize=(14,6))
 
# Plotting the central waveform

ax.plot(central_waveform['normalized_time'], central_waveform[f'normalized_{spg_col_name[0]}'], 'C0', label='SPG Waveform', linewidth = 2)
 
# Creating a second y-axis for the derivatives

ax2 = ax.twinx()
 
# Plotting the first, second, third derivatives

ax2.plot(central_waveform['normalized_time'], central_waveform[f'normalized_{spg_col_name[0]}_1_derivative']/central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].max(), 'C1', label='1st Derivative', linewidth = 2)

ax2.plot(central_waveform['normalized_time'], central_waveform[f'normalized_{spg_col_name[0]}_2_derivative']/central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].max(), 'C2', label='2nd Derivative', linewidth = 2)

# ax2.plot(central_waveform['normalized_time'], central_waveform[f'normalized_{spg_col_name[0]}_3_derivative']/central_waveform[f'normalized_{spg_col_name[0]}_3_derivative'].max(), 'C3', label='3rd Derivative')

def find_peaks_and_troughs(y):
    # Calculate the amplitude range
    amp_range = np.max(y) - np.min(y)
    
    # Set a lower prominence threshold
    prominence = 0.005 * amp_range  # Reduced from 0.01 to 0.005
    
    # Set a smaller distance to detect closer peaks
    distance = len(y) // 10  # Changed from len(y) // 4 to len(y) // 10
    
    # Find peaks and troughs
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-y, distance=distance, prominence=prominence)
    
    return peaks, troughs

# Find peaks and troughs for each waveform
spg_peaks, spg_troughs = find_peaks_and_troughs(central_waveform[f'normalized_{spg_col_name[0]}'])
deriv1_peaks, deriv1_troughs = find_peaks_and_troughs(central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'])
deriv2_peaks, deriv2_troughs = find_peaks_and_troughs(central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'])


print(f"SPG peaks: {spg_peaks}, troughs: {spg_troughs}")
print(f"1st derivative peaks: {deriv1_peaks}, troughs: {deriv1_troughs}")
print(f"2nd derivative peaks: {deriv2_peaks}, troughs: {deriv2_troughs}")
print(f"Length of central_waveform: {len(central_waveform)}")
print(f"Column names in central_waveform: {central_waveform.columns}")
# Annotate SPG waveform
if len(spg_peaks) > 0:
    ax.text(central_waveform['normalized_time'].iloc[spg_peaks[0]], 
            central_waveform[f'normalized_{spg_col_name[0]}'].iloc[spg_peaks[0]], 
            'S', ha='center', va='bottom')
if len(spg_troughs) > 0:
    ax.text(central_waveform['normalized_time'].iloc[spg_troughs[0]], 
            central_waveform[f'normalized_{spg_col_name[0]}'].iloc[spg_troughs[0]], 
            'N', ha='center', va='top')
if len(spg_peaks) > 1:
    ax.text(central_waveform['normalized_time'].iloc[spg_peaks[1]], 
            central_waveform[f'normalized_{spg_col_name[0]}'].iloc[spg_peaks[1]], 
            'D', ha='center', va='bottom')

# Annotate 1st derivative
if len(deriv1_peaks) > 0:
    ax2.text(central_waveform['normalized_time'].iloc[deriv1_peaks[0]], 
             central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].iloc[deriv1_peaks[0]]/central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].max(), 
             'w', ha='center', va='bottom')
if len(deriv1_troughs) > 0:
    ax2.text(central_waveform['normalized_time'].iloc[deriv1_troughs[0]], 
             central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].iloc[deriv1_troughs[0]]/central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].max(), 
             'y', ha='center', va='top')
if len(deriv1_peaks) > 1:
    ax2.text(central_waveform['normalized_time'].iloc[deriv1_peaks[1]], 
             central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].iloc[deriv1_peaks[1]]/central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].max(), 
             'z', ha='center', va='bottom')

# Annotate 2nd derivative
labels = ['a', 'b', 'c', 'd', 'e']
for i, peak in enumerate(deriv2_peaks[:3]):
    ax2.text(central_waveform['normalized_time'].iloc[peak], 
             central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].iloc[peak]/central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].max(), 
             labels[i*2], ha='center', va='bottom')
for i, trough in enumerate(deriv2_troughs[:2]):
    ax2.text(central_waveform['normalized_time'].iloc[trough], 
             central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].iloc[trough]/central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].max(), 
             labels[i*2+1], ha='center', va='top')


# Find intersection points between 1st and 2nd derivatives
def find_intersections(x, y1, y2):
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    return sign_changes

# Normalize derivatives for comparison
norm_deriv1 = central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'] / central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].max()
norm_deriv2 = central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'] / central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].max()

intersections = find_intersections(central_waveform['normalized_time'], norm_deriv1, norm_deriv2)

if len(intersections) >= 2:
    x_index = intersections[1]  
    x_time = central_waveform['normalized_time'].iloc[x_index]
    x_value = norm_deriv1.iloc[x_index]  

    ax2.plot(x_time, x_value, 'ko', markersize=6)
    # Annotate x
    ax2.annotate('x', (x_time, x_value), textcoords="offset points", xytext=(0,10), ha='center')
    
    print(f"Second intersection point 'x' found at time: {x_time}, value: {x_value}")
else:
    print("Not enough intersection points found between 1st and 2nd derivatives")



# Setting labels and title

ax.set_xlabel('Time [s]', fontweight = 'bold')

ax.set_ylabel('Normalized SPG Amplitude [a.u.]', fontweight = 'bold')

ax2.set_ylabel('Derivative Amplitude [a.u.]', fontweight = 'bold')

plt.title('Normalized SPG Waveform and its Derivatives', fontweight = 'bold')
 
# Adding legends for both axes

fig.legend(bbox_to_anchor=(0.8, 0.8), loc='upper right', fontsize=14, ncol=1)
plt.tight_layout()

plt.show()


def calculate_timespans(central_waveform, three_waveforms, end_of_central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs):
    timespans_dict = {}
    try:
        # Extract next waveform
        next_waveform_start = end_of_central_waveform + 1
        next_waveform = three_waveforms.loc[next_waveform_start:]
        
        # Extract relevant times for current waveform
        time = central_waveform['normalized_time']
        not_normal_time = central_waveform['time']

        # Extract times for current waveform
        S_time = time.iloc[spg_peaks[0]] if spg_peaks.size > 0 else None
        N_time = time.iloc[spg_troughs[0]] if spg_troughs.size > 0 else None
        D_time = time.iloc[spg_peaks[1]] if spg_peaks.size > 1 else None
        w_time = time.iloc[deriv1_peaks[0]] if deriv1_peaks.size > 0 else None
        y_time = time.iloc[deriv1_troughs[0]] if deriv1_troughs.size > 0 else None
        z_time = time.iloc[deriv1_peaks[1]] if deriv1_peaks.size > 1 else None
        a_time = time.iloc[deriv2_peaks[0]] if deriv2_peaks.size > 0 else None
        b_time = time.iloc[deriv2_troughs[0]] if deriv2_troughs.size > 0 else None
        c_time = time.iloc[deriv2_peaks[1]] if deriv2_peaks.size > 1 else None
        d_time = time.iloc[deriv2_troughs[1]] if deriv2_troughs.size > 1 else None
        e_time = time.iloc[deriv2_peaks[2]] if deriv2_peaks.size > 2 else None

        w_notnormal_time = not_normal_time.iloc[deriv1_peaks[0]] if deriv1_peaks.size > 0 else None
        a_notnormal_time = not_normal_time.iloc[deriv2_peaks[0]] if deriv2_peaks.size > 0 else None
        b_notnormal_time = not_normal_time.iloc[deriv2_troughs[0]] if deriv2_troughs.size > 0 else None
        N_notnormal_time = not_normal_time.iloc[spg_troughs[0]] if spg_troughs.size > 0 else None

        # Initialize next waveform times
        next_w_notnormal_time = None
        next_a_notnormal_time = None
        next_b_notnormal_time = None
        next_N_notnormal_time = None

        if not next_waveform.empty:
            # Extract relevant times for next waveform
            next_time = next_waveform['time']
            first_derivative_col = f'normalized_{spg_col_name[0]}_1_derivative'
            second_derivative_col = f'normalized_{spg_col_name[0]}_2_derivative'

            if first_derivative_col in next_waveform.columns:
                next_deriv1_peaks, _ = find_peaks(next_waveform[first_derivative_col])
                next_w_notnormal_time = next_time.iloc[next_deriv1_peaks[0]] if len(next_deriv1_peaks) > 0 else None

            if second_derivative_col in next_waveform.columns:
                next_deriv2_peaks, _ = find_peaks(next_waveform[second_derivative_col])
                next_deriv2_troughs, _ = find_peaks(-next_waveform[second_derivative_col])
                next_a_notnormal_time = next_time.iloc[next_deriv2_peaks[0]] if len(next_deriv2_peaks) > 0 else None
                next_b_notnormal_time = next_time.iloc[next_deriv2_troughs[0]] if len(next_deriv2_troughs) > 0 else None

            # Find next N time
            next_spg_troughs, _ = find_peaks(-next_waveform[spg_col_name[0]])
            next_N_notnormal_time = next_time.iloc[next_spg_troughs[0]] if len(next_spg_troughs) > 0 else None

        # Calculate inter-waveform timespans using non-normalized time
        timespan_ss = next_w_notnormal_time - w_notnormal_time if w_notnormal_time is not None and next_w_notnormal_time is not None else None
        timespan_aa2 = next_a_notnormal_time - a_notnormal_time if a_notnormal_time is not None and next_a_notnormal_time is not None else None
        timespan_bb2 = next_b_notnormal_time - b_notnormal_time if b_notnormal_time is not None and next_b_notnormal_time is not None else None
        

        # New calculations
        
        heart_rate = 60 / timespan_aa2 if timespan_aa2 is not None and timespan_aa2 != 0 else None

        # Find x_time (intersection of 1st and 2nd derivatives)
        norm_deriv1 = central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'] / central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'].max()
        norm_deriv2 = central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'] / central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'].max()
        intersections = find_intersections(central_waveform['normalized_time'], norm_deriv1, norm_deriv2)
        x_time = time.iloc[intersections[1]] if len(intersections) >= 2 else None

        # Calculate timespans
        O_time = time.iloc[0]
        timespan_S_O = S_time - O_time if S_time is not None else None
        timespan_D_O = D_time - O_time if D_time is not None else None
        timespan_N_O = N_time - O_time if N_time is not None else None
        timespan_w_O = w_time - O_time if w_time is not None else None
        timespan_y_O = y_time - O_time if y_time is not None else None
        timespan_z_O = z_time - O_time if z_time is not None else None
        timespan_x_O = x_time - O_time if x_time is not None else None
        timespan_a_O = a_time - O_time if a_time is not None else None
        timespan_b_O = b_time - O_time if b_time is not None else None
        timespan_c_O = c_time - O_time if c_time is not None else None
        timespan_d_O = d_time - O_time if d_time is not None else None
        timespan_e_O = e_time - O_time if e_time is not None else None
        timespan_aa2_N = timespan_aa2 - timespan_N_O if timespan_aa2 is not None and timespan_N_O is not None else None

        timespan_S_N = S_time - N_time if S_time is not None and N_time is not None else None
        timespan_D_N = D_time - N_time if D_time is not None and N_time is not None else None
        timespan_N_D = D_time - N_time if D_time is not None and N_time is not None else None
        
        timespan_w_y = y_time - w_time if y_time is not None and w_time is not None else None
        timespan_y_z = z_time - y_time if z_time is not None and y_time is not None else None
        
        timespan_a_b = b_time - a_time if b_time is not None and a_time is not None else None
        timespan_b_c = c_time - b_time if c_time is not None and b_time is not None else None
        timespan_c_d = d_time - c_time if d_time is not None and c_time is not None else None
        timespan_d_e = e_time - d_time if e_time is not None and d_time is not None else None

        timespan_c_S = c_time - S_time if c_time is not None and S_time is not None else None
        timespan_d_S = d_time - S_time if d_time is not None and S_time is not None else None
        timespan_e_S = e_time - S_time if e_time is not None and S_time is not None else None
        timespan_D_S = D_time - S_time if D_time is not None and S_time is not None else None

        timespan_c_b = c_time - b_time if c_time is not None and b_time is not None else None
        timespan_d_b = d_time - b_time if d_time is not None and b_time is not None else None
        timespan_b_w = b_time - w_time if b_time is not None and w_time is not None else None
        timespan_S_w = S_time - w_time if S_time is not None and w_time is not None else None
        timespan_c_w = c_time - w_time if c_time is not None and w_time is not None else None
        timespan_d_w = d_time - w_time if d_time is not None and w_time is not None else None
        timespan_z_w = z_time - w_time if z_time is not None and w_time is not None else None
        timespan_c_a = c_time - a_time if c_time is not None and a_time is not None else None
        timespan_b_a = b_time - a_time if b_time is not None and a_time is not None else None

        # Calculate timespan ratios
        tm_a_tm_ss = timespan_a_O / timespan_ss if timespan_a_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_w_tm_ss = timespan_w_O / timespan_ss if timespan_w_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_b_tm_ss = timespan_b_O / timespan_ss if timespan_b_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_S_tm_ss = timespan_S_O / timespan_ss if timespan_S_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_c_tm_ss = timespan_c_O / timespan_ss if timespan_c_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_y_tm_ss = timespan_y_O / timespan_ss if timespan_y_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_N_tm_ss = timespan_N_O / timespan_ss if timespan_N_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_z_tm_ss = timespan_z_O / timespan_ss if timespan_z_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_x_tm_ss = timespan_x_O / timespan_ss if timespan_x_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_D_tm_ss = timespan_D_O / timespan_ss if timespan_D_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_e_tm_ss = timespan_e_O / timespan_ss if timespan_e_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_wz_tm_ss = (timespan_w_O * timespan_z_O) / timespan_ss if timespan_w_O is not None and timespan_z_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_SD_tm_ss = (timespan_S_O * timespan_D_O) / timespan_ss if timespan_S_O is not None and timespan_D_O is not None and timespan_ss is not None and timespan_ss != 0 else None
        tm_bb2_tm_ss = timespan_b_O / timespan_ss if timespan_b_O is not None and timespan_ss is not None and timespan_ss != 0 else None

        tm_S_tm_N = timespan_S_O / timespan_N_O if timespan_S_O is not None and timespan_N_O is not None and timespan_N_O != 0 else None
  

     # Create a dictionary with all calculated values
        timespans_dict = {
            'timespan_ss': timespan_ss,
            'timespan_aa2': timespan_aa2,
            'timespan_bb2': timespan_bb2,
            'timespan_aa2_N': timespan_aa2_N,
            'heart_rate': heart_rate,
            'timespan_S_O': timespan_S_O,
            'timespan_D_O': timespan_D_O,
            'timespan_N_O': timespan_N_O,
            'timespan_w_O': timespan_w_O,
            'timespan_y_O': timespan_y_O,
            'timespan_z_O': timespan_z_O,
            'timespan_x_O': timespan_x_O,
            'timespan_a_O': timespan_a_O,
            'timespan_b_O': timespan_b_O,
            'timespan_c_O': timespan_c_O,
            'timespan_d_O': timespan_d_O,
            'timespan_e_O': timespan_e_O,
            'tm_a_tm_ss': tm_a_tm_ss,
            'tm_w_tm_ss': tm_w_tm_ss,
            'tm_b_tm_ss': tm_b_tm_ss,
            'tm_S_tm_ss': tm_S_tm_ss,
            'tm_c_tm_ss': tm_c_tm_ss,
            'tm_y_tm_ss': tm_y_tm_ss,
            'tm_N_tm_ss': tm_N_tm_ss,
            'tm_z_tm_ss': tm_z_tm_ss,
            'tm_x_tm_ss': tm_x_tm_ss,
            'tm_D_tm_ss': tm_D_tm_ss,
            'tm_e_tm_ss': tm_e_tm_ss,
            'tm_wz_tm_ss': tm_wz_tm_ss,
            'tm_SD_tm_ss': tm_SD_tm_ss,
            'tm_bb2_tm_ss': tm_bb2_tm_ss,
            'tm_S_tm_N': tm_S_tm_N  # New ratio
        }
    except Exception as e:
        print(f"Error in calculate_timespans: {str(e)}")
        
    return timespans_dict

def calculate_amplitudes(central_waveform, three_waveforms, end_of_central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs):
    amplitudes = {}
    try:
        spg_signal = central_waveform[f'normalized_{spg_col_name[0]}']
        vpg_signal = central_waveform[f'normalized_{spg_col_name[0]}_1_derivative']
        apg_signal = central_waveform[f'normalized_{spg_col_name[0]}_2_derivative']

        # SPG amplitudes
        Am_O = spg_signal.iloc[0]
        # Calculate Am_O2
        next_waveform_start = end_of_central_waveform + 1
        next_waveform = three_waveforms.loc[next_waveform_start:]
        Am_O2 = next_waveform[f'normalized_{spg_col_name[0]}'].iloc[0] - Am_O if not next_waveform.empty else None
        
        Am_S = spg_signal.iloc[spg_peaks[0]] - Am_O if len(spg_peaks) > 0 else None
        Am_N = spg_signal.iloc[spg_troughs[0]] - Am_O if len(spg_troughs) > 0 else None
        Am_D = spg_signal.iloc[spg_peaks[1]] - Am_O if len(spg_peaks) > 1 else None
        
        # VPG amplitudes
        Am_w = vpg_signal.iloc[deriv1_peaks[0]] - Am_O if len(deriv1_peaks) > 0 else None
        Am_y = vpg_signal.iloc[deriv1_troughs[0]] - Am_O if len(deriv1_troughs) > 0 else None

        # APG amplitudes
        Am_a = apg_signal.iloc[deriv2_peaks[0]] if len(deriv2_peaks) > 0 else None
        Am_b = apg_signal.iloc[deriv2_troughs[0]] if len(deriv2_troughs) > 0 else None
        Am_c = apg_signal.iloc[deriv2_peaks[1]] if len(deriv2_peaks) > 1 else None
        Am_d = apg_signal.iloc[deriv2_troughs[1]] if len(deriv2_troughs) > 1 else None
        Am_e = apg_signal.iloc[deriv2_peaks[2]] if len(deriv2_peaks) > 2 else None

        # Amplitude differences
        Am_S_O = Am_S
        Am_N_O = Am_N
        Am_D_O = Am_D
        Am_w_O = Am_w
        Am_y_O = Am_y
        Am_a_O = Am_a - Am_O if Am_a is not None else None
        Am_b_O = Am_b - Am_O if Am_b is not None else None
        Am_c_O = Am_c - Am_O if Am_c is not None else None
        
        Am_N_S = Am_S_O - Am_N_O if Am_S_O is not None and Am_N_O is not None else None

        # Amplitude ratios
        def safe_divide(a, b):
            if a is None or b is None or b == 0:
                return None
            return a / b

        Am_N_Am_S = safe_divide(Am_N_O, Am_S_O)
        Am_D_Am_S = safe_divide(Am_D_O, Am_S_O)
        Am_NS_Am_S = safe_divide(Am_N_S, Am_S_O)
        Am_DS_Am_S = safe_divide(Am_S_O - Am_D_O, Am_S_O) if Am_S_O is not None and Am_D_O is not None else None
        
        # New amplitude ratios
        Am_a_Am_S = safe_divide(Am_a, Am_S_O)
        Am_w_Am_S = safe_divide(Am_w_O, Am_S_O)
        Am_b_Am_S = safe_divide(Am_b, Am_S_O)
        Am_c_Am_S = safe_divide(Am_c, Am_S_O)
        Am_y_Am_S = safe_divide(Am_y_O, Am_S_O)
        Am_O2_Am_S = safe_divide(Am_O2, Am_S_O)
        Am_S_Am_O = safe_divide(Am_S_O, Am_O)

        amplitudes = {
            'Am_S_O': Am_S_O,
            'Am_N_O': Am_N_O,
            'Am_D_O': Am_D_O,
            'Am_w_O': Am_w_O,
            'Am_y_O': Am_y_O,
            'Am_a_O': Am_a_O,
            'Am_b_O': Am_b_O,
            'Am_c_O': Am_c_O,
            'Am_N_S': Am_N_S,
            'Am_N_Am_S': Am_N_Am_S,
            'Am_D_Am_S': Am_D_Am_S,
            'Am_NS_Am_S': Am_NS_Am_S,
            'Am_DS_Am_S': Am_DS_Am_S,
            'Am_a_Am_S': Am_a_Am_S,  
            'Am_w_Am_S': Am_w_Am_S,  
            'Am_b_Am_S': Am_b_Am_S,  
            'Am_c_Am_S': Am_c_Am_S,  
            'Am_y_Am_S': Am_y_Am_S,  
            'Am_O2_Am_S': Am_O2_Am_S,
            'Am_S_Am_O': Am_S_Am_O,
        }
    except Exception as e:
        print(f"Error in calculate_amplitudes: {str(e)}")
    
    return amplitudes
def calculate_areas(central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv2_peaks, deriv2_troughs):
    spg_signal = central_waveform[f'normalized_{spg_col_name[0]}']
    vpg_signal = central_waveform[f'normalized_{spg_col_name[0]}_1_derivative']
    apg_signal = central_waveform[f'normalized_{spg_col_name[0]}_2_derivative']
    tpg_signal = central_waveform[f'normalized_{spg_col_name[0]}_3_derivative']
    time = central_waveform['normalized_time']

    index_S = spg_peaks[0] if len(spg_peaks) > 0 else None
    index_N = spg_troughs[0] if len(spg_troughs) > 0 else None
    index_w = deriv1_peaks[0] if len(deriv1_peaks) > 0 else None
    index_c = deriv2_peaks[1] if len(deriv2_peaks) > 1 else None
    index_d = deriv2_troughs[1] if len(deriv2_troughs) > 1 else None

    # Calculate areas for the entire waveform
    area_signal = np.trapz(spg_signal, time)
    area_first_derivative = np.trapz(vpg_signal, time)
    area_second_derivative = np.trapz(apg_signal, time)
    area_third_derivative = np.trapz(tpg_signal, time)

    # New area calculations
    ar_OO = area_signal  # This is the same as area_signal
    ar_OS = np.trapz(spg_signal[:index_S+1], time[:index_S+1]) if index_S is not None else None
    ar_Oc = np.trapz(spg_signal[:index_c+1], time[:index_c+1]) if index_c is not None else None
    ar_ON = np.trapz(spg_signal[:index_N+1], time[:index_N+1]) if index_N is not None else None
    
    # Calculate ar_NO2
    if index_N is not None:
        next_waveform_start = end_of_central_waveform + 1
        next_waveform = three_waveforms.loc[next_waveform_start:]
        if not next_waveform.empty:
            next_time = next_waveform['normalized_time']
            next_signal = next_waveform[f'normalized_{spg_col_name[0]}']
            combined_signal = pd.concat([spg_signal[index_N:], next_signal.iloc[:1]])
            combined_time = pd.concat([time[index_N:], next_time.iloc[:1]])
            ar_NO2 = np.trapz(combined_signal, combined_time)
        else:
            ar_NO2 = None
    else:
        ar_NO2 = None
    # Calculate IPA ratio
    IPA_ratio = ar_NO2 / ar_ON if ar_NO2 is not None and ar_ON is not None and ar_ON != 0 else None
    power_area_vpg = np.trapz(vpg_signal**2, time)
    power_area_apg = np.trapz(apg_signal**2, time)

    # Calculate power areas for different segments
    if all(index is not None for index in [index_S, index_w, index_c, index_d]):
        time_slice_O_S = time[:index_S+1]
        time_slice_w_S = time[index_w:index_S+1]
        time_slice_S_c = time[index_S:index_c+1]
        time_slice_S_d = time[index_S:index_d+1]

        # SPG power areas
        spg_power_O_S = spg_signal[:index_S+1]**2
        spg_power_w_S = spg_signal[index_w:index_S+1]**2
        spg_power_S_c = spg_signal[index_S:index_c+1]**2
        spg_power_S_d = spg_signal[index_S:index_d+1]**2

        power_area_O_S = np.trapz(spg_power_O_S, time_slice_O_S)
        power_area_w_S = np.trapz(spg_power_w_S, time_slice_w_S)
        power_area_S_c = np.trapz(spg_power_S_c, time_slice_S_c)
        power_area_S_d = np.trapz(spg_power_S_d, time_slice_S_d)

        # VPG power areas
        vpg_power_O_S = vpg_signal[:index_S+1]**2
        vpg_power_w_S = vpg_signal[index_w:index_S+1]**2
        vpg_power_S_c = vpg_signal[index_S:index_c+1]**2
        vpg_power_S_d = vpg_signal[index_S:index_d+1]**2

        vpg_power_area_O_S = np.trapz(vpg_power_O_S, time_slice_O_S)
        vpg_power_area_w_S = np.trapz(vpg_power_w_S, time_slice_w_S)
        vpg_power_area_S_c = np.trapz(vpg_power_S_c, time_slice_S_c)
        vpg_power_area_S_d = np.trapz(vpg_power_S_d, time_slice_S_d)

        # APG power areas
        apg_power_O_S = apg_signal[:index_S+1]**2
        apg_power_w_S = apg_signal[index_w:index_S+1]**2
        apg_power_S_c = apg_signal[index_S:index_c+1]**2
        apg_power_S_d = apg_signal[index_S:index_d+1]**2

        apg_power_area_O_S = np.trapz(apg_power_O_S, time_slice_O_S)
        apg_power_area_w_S = np.trapz(apg_power_w_S, time_slice_w_S)
        apg_power_area_S_c = np.trapz(apg_power_S_c, time_slice_S_c)
        apg_power_area_S_d = np.trapz(apg_power_S_d, time_slice_S_d)

        # Calculate power area ratios
        power_area_O_O = np.trapz(spg_signal**2, time)  # Total power area for SPG
        vpg_power_area_O_O = np.trapz(vpg_signal**2, time)  # Total power area for VPG
        apg_power_area_O_O = np.trapz(apg_signal**2, time)  # Total power area for APG

        # SPG ratios
        ratio_PA_OS_spg = power_area_O_S / power_area_O_O if power_area_O_O != 0 else None
        ratio_PA_wS_spg = power_area_w_S / power_area_O_O if power_area_O_O != 0 else None
        ratio_PA_Sc_spg = power_area_S_c / power_area_O_O if power_area_O_O != 0 else None
        ratio_PA_Sd_spg = power_area_S_d / power_area_O_O if power_area_O_O != 0 else None

        # VPG ratios
        ratio_PA_OS_vpg = vpg_power_area_O_S / vpg_power_area_O_O if vpg_power_area_O_O != 0 else None
        ratio_PA_wS_vpg = vpg_power_area_w_S / vpg_power_area_O_O if vpg_power_area_O_O != 0 else None
        ratio_PA_Sc_vpg = vpg_power_area_S_c / vpg_power_area_O_O if vpg_power_area_O_O != 0 else None
        ratio_PA_Sd_vpg = vpg_power_area_S_d / vpg_power_area_O_O if vpg_power_area_O_O != 0 else None

        # APG ratios
        ratio_PA_OS_apg = apg_power_area_O_S / apg_power_area_O_O if apg_power_area_O_O != 0 else None
        ratio_PA_wS_apg = apg_power_area_w_S / apg_power_area_O_O if apg_power_area_O_O != 0 else None
        ratio_PA_Sc_apg = apg_power_area_S_c / apg_power_area_O_O if apg_power_area_O_O != 0 else None
        ratio_PA_Sd_apg = apg_power_area_S_d / apg_power_area_O_O if apg_power_area_O_O != 0 else None

    else:
        power_area_O_S = power_area_w_S = power_area_S_c = power_area_S_d = None
        vpg_power_area_O_S = vpg_power_area_w_S = vpg_power_area_S_c = vpg_power_area_S_d = None
        apg_power_area_O_S = apg_power_area_w_S = apg_power_area_S_c = apg_power_area_S_d = None
        ratio_PA_OS_spg = ratio_PA_wS_spg = ratio_PA_Sc_spg = ratio_PA_Sd_spg = None
        ratio_PA_OS_vpg = ratio_PA_wS_vpg = ratio_PA_Sc_vpg = ratio_PA_Sd_vpg = None
        ratio_PA_OS_apg = ratio_PA_wS_apg = ratio_PA_Sc_apg = ratio_PA_Sd_apg = None

    return {
        'ar_OO': ar_OO,
        'ar_OS': ar_OS,
        'ar_Oc': ar_Oc,
        'ar_ON': ar_ON,
        'power_area_vpg': power_area_vpg,
        'power_area_apg': power_area_apg,
        'power_area_O_S': power_area_O_S,
        'power_area_w_S': power_area_w_S,
        'power_area_S_c': power_area_S_c,
        'power_area_S_d': power_area_S_d,
        'vpg_power_area_O_S': vpg_power_area_O_S,
        'vpg_power_area_w_S': vpg_power_area_w_S,
        'vpg_power_area_S_c': vpg_power_area_S_c,
        'vpg_power_area_S_d': vpg_power_area_S_d,
        'apg_power_area_O_S': apg_power_area_O_S,
        'apg_power_area_w_S': apg_power_area_w_S,
        'apg_power_area_S_c': apg_power_area_S_c,
        'apg_power_area_S_d': apg_power_area_S_d,
        'ratio_PA_OS_spg': ratio_PA_OS_spg,
        'ratio_PA_wS_spg': ratio_PA_wS_spg,
        'ratio_PA_Sc_spg': ratio_PA_Sc_spg,
        'ratio_PA_Sd_spg': ratio_PA_Sd_spg,
        'ratio_PA_OS_vpg': ratio_PA_OS_vpg,
        'ratio_PA_wS_vpg': ratio_PA_wS_vpg,
        'ratio_PA_Sc_vpg': ratio_PA_Sc_vpg,
        'ratio_PA_Sd_vpg': ratio_PA_Sd_vpg,
        'ratio_PA_OS_apg': ratio_PA_OS_apg,
        'ratio_PA_wS_apg': ratio_PA_wS_apg,
        'ratio_PA_Sc_apg': ratio_PA_Sc_apg,
        'ratio_PA_Sd_apg': ratio_PA_Sd_apg,
        'IPA ratio': IPA_ratio
    }


def calculate_slopes(central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs):
    spg_signal = central_waveform[f'normalized_{spg_col_name[0]}']
    vpg_signal = central_waveform[f'normalized_{spg_col_name[0]}_1_derivative']
    apg_signal = central_waveform[f'normalized_{spg_col_name[0]}_2_derivative']
    time = central_waveform['normalized_time']
    
    sample_time = 0.001  # Assuming the sample time is 1ms, adjust if different

    # Extract relevant times and amplitudes
    O_time, O_amp = time.iloc[0], spg_signal.iloc[0]
    S_time, S_amp = (time.iloc[spg_peaks[0]], spg_signal.iloc[spg_peaks[0]]) if len(spg_peaks) > 0 else (None, None)
    N_time, N_amp = (time.iloc[spg_troughs[0]], spg_signal.iloc[spg_troughs[0]]) if len(spg_troughs) > 0 else (None, None)
    D_time, D_amp = (time.iloc[spg_peaks[1]], spg_signal.iloc[spg_peaks[1]]) if len(spg_peaks) > 1 else (None, None)
    w_time, w_amp = (time.iloc[deriv1_peaks[0]], spg_signal.iloc[deriv1_peaks[0]]) if len(deriv1_peaks) > 0 else (None, None)
    y_time, y_amp = (time.iloc[deriv1_troughs[0]], spg_signal.iloc[deriv1_troughs[0]]) if len(deriv1_troughs) > 0 else (None, None)
    a_time, a_amp = (time.iloc[deriv2_peaks[0]], spg_signal.iloc[deriv2_peaks[0]]) if len(deriv2_peaks) > 0 else (None, None)
    b_time, b_amp = (time.iloc[deriv2_troughs[0]], spg_signal.iloc[deriv2_troughs[0]]) if len(deriv2_troughs) > 0 else (None, None)
    c_time, c_amp = (time.iloc[deriv2_peaks[1]], spg_signal.iloc[deriv2_peaks[1]]) if len(deriv2_peaks) > 1 else (None, None)
    d_time, d_amp = (time.iloc[deriv2_troughs[1]], spg_signal.iloc[deriv2_troughs[1]]) if len(deriv2_troughs) > 1 else (None, None)
    e_time, e_amp = (time.iloc[deriv2_peaks[2]], spg_signal.iloc[deriv2_peaks[2]]) if len(deriv2_peaks) > 2 else (None, None)

    # APG amplitudes
    apg_O_amp = apg_signal.iloc[0]
    apg_w_amp = apg_signal.iloc[deriv1_peaks[0]] if len(deriv1_peaks) > 0 else None
    apg_S_amp = apg_signal.iloc[spg_peaks[0]] if len(spg_peaks) > 0 else None
    apg_a_amp = apg_signal.iloc[deriv2_peaks[0]] if len(deriv2_peaks) > 0 else None
    apg_b_amp = apg_signal.iloc[deriv2_troughs[0]] if len(deriv2_troughs) > 0 else None
    apg_c_amp = apg_signal.iloc[deriv2_peaks[1]] if len(deriv2_peaks) > 1 else None
    apg_d_amp = apg_signal.iloc[deriv2_troughs[1]] if len(deriv2_troughs) > 1 else None
    apg_e_amp = apg_signal.iloc[deriv2_peaks[2]] if len(deriv2_peaks) > 2 else None

    # Calculate slopes
    def safe_slope(y2, y1, x2, x1):
        if all(v is not None for v in [y2, y1, x2, x1]) and x2 != x1:
            return (y2 - y1) / ((x2 - x1) * sample_time)
        return None

    # SPG slopes
    spg_SL_S_c = safe_slope(c_amp, S_amp, c_time, S_time)
    spg_SL_S_d = safe_slope(d_amp, S_amp, d_time, S_time)
    spg_SL_b_S = safe_slope(S_amp, b_amp, S_time, b_time)
    spg_SL_b_c = safe_slope(c_amp, b_amp, c_time, b_time)
    spg_SL_b_d = safe_slope(d_amp, b_amp, d_time, b_time)
    spg_SL_w_S = safe_slope(S_amp, w_amp, S_time, w_time)
    spg_SL_O_S = safe_slope(S_amp, O_amp, S_time, O_time)
    spg_SL_a_b = safe_slope(b_amp, a_amp, b_time, a_time)

    # APG slopes
    apg_SL_a_b = safe_slope(apg_b_amp, apg_a_amp, b_time, a_time)
    apg_SL_b_S = safe_slope(apg_S_amp, apg_b_amp, S_time, b_time)
    apg_SL_b_c = safe_slope(apg_c_amp, apg_b_amp, c_time, b_time)
    apg_SL_b_d = safe_slope(apg_d_amp, apg_b_amp, d_time, b_time)
    apg_SL_b_e = safe_slope(apg_e_amp, apg_b_amp, e_time, b_time)
    apg_SL_S_c = safe_slope(apg_c_amp, apg_S_amp, c_time, S_time)
    apg_SL_w_S = safe_slope(apg_S_amp, apg_w_amp, S_time, w_time)
    apg_SL_O_S = safe_slope(apg_S_amp, apg_O_amp, S_time, O_time)

    return {
        'spg_SL_S_c': spg_SL_S_c,
        'spg_SL_S_d': spg_SL_S_d,
        'spg_SL_b_S': spg_SL_b_S,
        'spg_SL_b_c': spg_SL_b_c,
        'spg_SL_b_d': spg_SL_b_d,
        'spg_SL_w_S': spg_SL_w_S,
        'spg_SL_O_S': spg_SL_O_S,
        'spg_SL_a_b': spg_SL_a_b,
        'apg_SL_a_b': apg_SL_a_b,
        'apg_SL_b_S': apg_SL_b_S,
        'apg_SL_b_c': apg_SL_b_c,
        'apg_SL_b_d': apg_SL_b_d,
        'apg_SL_b_e': apg_SL_b_e,
        'apg_SL_S_c': apg_SL_S_c,
        'apg_SL_w_S': apg_SL_w_S,
        'apg_SL_O_S': apg_SL_O_S,
    }
def calculate_ratios(central_waveform, spg_col_name, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs):
    vpg_signal = central_waveform[f'normalized_{spg_col_name[0]}_1_derivative']
    apg_signal = central_waveform[f'normalized_{spg_col_name[0]}_2_derivative']

    # VPG ratios
    vpg_z = vpg_signal.iloc[deriv1_peaks[1]] if len(deriv1_peaks) > 1 else None
    vpg_w = vpg_signal.iloc[deriv1_peaks[0]]
    vpg_y = vpg_signal.iloc[deriv1_troughs[0]]
    
    # New additions for cc_w and dd_w
    vpg_c = vpg_signal.iloc[deriv2_peaks[1]] if len(deriv2_peaks) > 1 else None
    vpg_d = vpg_signal.iloc[deriv2_troughs[1]] if len(deriv2_troughs) > 1 else None

    z_w = vpg_z / vpg_w if vpg_z is not None else None
    y_w = vpg_y / vpg_w
    
    # New ratios
    cc_w = vpg_c / vpg_w if vpg_c is not None else None
    dd_w = vpg_d / vpg_w if vpg_d is not None else None

    # APG ratios
    apg_a = apg_signal.iloc[deriv2_peaks[0]]
    apg_b = apg_signal.iloc[deriv2_troughs[0]]
    apg_c = apg_signal.iloc[deriv2_peaks[1]] if len(deriv2_peaks) > 1 else None
    apg_d = apg_signal.iloc[deriv2_troughs[1]] if len(deriv2_troughs) > 1 else None
    apg_e = apg_signal.iloc[deriv2_peaks[2]] if len(deriv2_peaks) > 2 else None

    b_a = apg_b / apg_a
    c_a = apg_c / apg_a if apg_c is not None else None
    d_a = apg_d / apg_a if apg_d is not None else None
    e_a = apg_e / apg_a if apg_e is not None else None

    agi = (apg_b - apg_c - apg_d - apg_e) / apg_a if all(v is not None for v in [apg_c, apg_d, apg_e]) else None
    agi_mod = (apg_b - apg_c - apg_d) / apg_a if all(v is not None for v in [apg_c, apg_d]) else None

    return {
        'z_w': z_w,
        'y_w': y_w,
        'cc_w': cc_w,  
        'dd_w': dd_w,  
        'b_a': b_a,
        'c_a': c_a,
        'd_a': d_a,
        'e_a': e_a,
        'agi': agi,
        'agi_mod': agi_mod,
    }

# After plotting and annotation code

# Calculate features
timespans = calculate_timespans(central_waveform, three_waveforms, end_of_central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
amplitudes = calculate_amplitudes(central_waveform, three_waveforms, end_of_central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
areas = calculate_areas(central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv2_peaks, deriv2_troughs)
slopes = calculate_slopes(central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
ratios = calculate_ratios(central_waveform, spg_col_name, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
print(timespans)
print("\n--- Timespans ---")
for key, value in timespans.items():
    print(f"{key}: {value}")

print("\n--- Amplitudes ---")
for key, value in amplitudes.items():
    print(f"{key}: {value}")

print("\n--- Areas ---")
for key, value in areas.items():
    print(f"{key}: {value}")

print("\n--- Slopes ---")
for key, value in slopes.items():
    print(f"{key}: {value}")

print("\n--- Ratios ---")
for key, value in ratios.items():
    print(f"{key}: {value}")
 