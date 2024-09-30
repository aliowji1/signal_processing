# SPG Analysis Project

This project performs analysis on Sphygmogram (SPG) data, including feature extraction, processing, regression analysis, and Principal Component Analysis (PCA).

## Files in this Repository

1. `extractfeatures.py`: Contains functions for extracting features from SPG waveforms.
2. `processing.py`: Processes raw patient data, extracts features, and calculates average features for each patient.
3. `regression.py`: Performs regression analysis on the average features and creates visualizations.
4. `PCA.py`: Conducts Principal Component Analysis on the extracted features.
5. `main_analysis.py`: Orchestrates the entire analysis pipeline by running the other scripts in sequence.

## How to Use

1. Ensure you have Python installed on your system, along with the required libraries (pandas, numpy, matplotlib, scipy, sklearn).

2. Clone this repository to your local machine:
   ```
   git clone https://github.com/yourusername/spg-analysis.git
   cd spg-analysis
   ```

3. Prepare your SPG data files and update the file paths in `processing.py` to point to your data.

4. Run the main analysis script:
   ```
   python main_analysis.py
   ```

   This will execute the entire analysis pipeline:
   - Extract features from the SPG waveforms
   - Process the data and calculate average features for each patient
   - Perform regression analysis
   - Conduct PCA on the features

5. Check the output directory (specified in the scripts) for results, including:
   - CSV files with extracted features and average patient data
   - Regression plots
   - PCA results and visualizations

## Customization

- You can modify the feature extraction process in `extractfeatures.py`
- Adjust the regression analysis parameters in `regression.py`
- Customize the PCA settings in `PCA.py`
- Update the workflow in `main_analysis.py` if you need to change the order of operations or add new steps

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

## Contributing

Feel free to fork this repository and submit pull requests with any enhancements.

## License

[Specify your license here, e.g., MIT, GPL, etc.]

## Contact

[Your Name]
[Your Email or other contact information]