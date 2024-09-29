import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# List of subject numbers
subjects = [1, 3, 4, 6, 12, 13, 14, 15, 16, 22, 23, 24, 28, 29, 30, 31, 35, 36]

# Base file path
base_path = "/Users/owjiaa/Desktop/rawdata2/"

# Iterate through each subject
for subject in subjects:
    file_path = f"{base_path}subject{subject}_annotated_waveforms_first280s_preocc.csv"
    
    print(f"Processing subject {subject}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Name of the column to normalize
    spg_col_name = ['spg_filt_upsampled']

    # Identify trough indices
    trough_indices = df.loc[df[f'{spg_col_name[0]}_troughs'] != 0].index

    # List to store features for all waveforms
    all_waveform_features = []

    # Iterate through all possible central waveforms
    for i in range(1, len(trough_indices) - 1):
        try:
            # Get 3 waveforms
            three_waveforms, start_of_central_waveform, end_of_central_waveform = get_3_SPGwaveforms(df, trough_indices, i)
            
            if three_waveforms is None:
                print(f"No valid waveforms found for index {i}, skipping.")
                continue

            # Normalize waveforms
            three_waveforms = three_waveforms.copy()
            scaler_amplitude = MinMaxScaler(feature_range=(0, 1))
            scaler_time = MinMaxScaler(feature_range=(0, 1))

            central_waveform_amplitude = three_waveforms.loc[start_of_central_waveform:end_of_central_waveform, f'{spg_col_name[0]}'].values.reshape(-1, 1)
            central_waveform_time = three_waveforms.loc[start_of_central_waveform:end_of_central_waveform, 'time'].values.reshape(-1, 1)

            scaler_amplitude.fit(central_waveform_amplitude)
            scaler_time.fit(central_waveform_time)

            three_waveforms[f'normalized_{spg_col_name[0]}'] = scaler_amplitude.transform(three_waveforms[[f'{spg_col_name[0]}']])
            three_waveforms['normalized_time'] = scaler_time.transform(three_waveforms[['time']])

            # Calculate derivatives
            three_waveforms = calculate_derivatives(three_waveforms, f'normalized_{spg_col_name[0]}', no_derivatives=4)

            # Extract central waveform
            central_waveform = extract_central_waveform(three_waveforms, start_of_central_waveform, end_of_central_waveform)

            # Find peaks and troughs
            spg_peaks, spg_troughs = find_peaks_and_troughs(central_waveform[f'normalized_{spg_col_name[0]}'])
            deriv1_peaks, deriv1_troughs = find_peaks_and_troughs(central_waveform[f'normalized_{spg_col_name[0]}_1_derivative'])
            deriv2_peaks, deriv2_troughs = find_peaks_and_troughs(central_waveform[f'normalized_{spg_col_name[0]}_2_derivative'])

            # Calculate features
            timespans = calculate_timespans(central_waveform, three_waveforms, end_of_central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
            amplitudes = calculate_amplitudes(central_waveform, three_waveforms, end_of_central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
            areas = calculate_areas(central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv2_peaks, deriv2_troughs)
            slopes = calculate_slopes(central_waveform, spg_col_name, spg_peaks, spg_troughs, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)
            ratios = calculate_ratios(central_waveform, spg_col_name, deriv1_peaks, deriv1_troughs, deriv2_peaks, deriv2_troughs)

            # Combine all features
            waveform_features = {
                'Subject': f'subject{subject}',
                'Waveform Index': i,
                **timespans,
                **amplitudes,
                **areas,
                **slopes,
                **ratios
            }

            all_waveform_features.append(waveform_features)

        except Exception as e:
            print(f"Error processing waveform {i} for subject {subject}: {str(e)}")

    # Convert list of dictionaries to DataFrame
    df_waveform_features = pd.DataFrame(all_waveform_features)

    # Save to CSV
    output_file = f'spgwaveform_features_subject{subject}.csv'
    df_waveform_features.to_csv(output_file, index=False)

    print(f"Waveform features for subject {subject} have been saved to '{output_file}'")

print("Processing complete for all subjects.")

# List of subject numbers
subjects = [1, 3, 12, 13, 14, 15, 16, 22, 23, 24, 29, 30, 31, 35, 36]

# List to store average features for each subject
subject_averages = []

directory = "/Users/owjiaa/Desktop/"

# Loop through each subject
for sub in subjects:
    # Construct the file path
    file_path = os.path.join(directory, f"spgwaveform_features_subject{sub}.csv")
    
    # Read the CSV file, converting problematic entries to NaN
    df = pd.read_csv(file_path, na_values=['null', 'error', '', ' '])
    
    # Calculate the mean of all numeric columns, ignoring NaN values
    subject_mean = df.mean(numeric_only=True, skipna=True)
    
    # Add subject identifier
    subject_mean['Subject'] = f"subject{sub}"
    
    # Append to the list of subject averages
    subject_averages.append(subject_mean)

# Combine all subject averages into a single dataframe
combined_averages = pd.DataFrame(subject_averages)

# Reorder columns to have 'Subject' as the first column
columns = ['Subject'] + [col for col in combined_averages.columns if col != 'Subject']
combined_averages = combined_averages[columns]

# Replace any remaining NaN values with a string indicating missing data
combined_averages = combined_averages.replace({np.nan: 'Missing'})

# Save the combined averages to a new CSV file
combined_averages.to_csv(os.path.join(directory, "newspg_average_features.csv"), index=False)

print("Average features for all subjects have been combined and saved to 'average_features.csv'")