import pandas as pd
import argparse
import glob

def compute_mean_and_std_across_files(file_paths):
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    
    # Check if all files have the same number of rows
    num_rows = [df.shape[0] for df in dfs]
    if len(set(num_rows)) != 1:
        print("Warning: Input files have different numbers of rows.")
    
    # Align all dataframes by filling missing values with NaN and concatenating them
    max_rows = max(num_rows)
    aligned_dfs = [df.reindex(range(max_rows)) for df in dfs]
    
    # Stack the dataframes and compute mean and std deviation row-wise
    combined_df = pd.concat(aligned_dfs, axis=0).groupby(level=0).agg(['mean', 'std'])
    
    # Prepare the final dataframe
    result_df = pd.DataFrame()
    result_df['Step'] = combined_df['Step']['mean']
    result_df['Mean-Value'] = combined_df['Value']['mean']
    result_df['Std-Dev'] = combined_df['Value']['std']
    
    return result_df

def compute_rolling_statistics(df, window_size):
    df['Rolling average'] = df['Mean-Value'].rolling(window=window_size).mean()
    df['Rolling std dev'] = df['Mean-Value'].rolling(window=window_size).std()
    return df[['Step', 'Rolling average', 'Rolling std dev']]

def main(input_folder, output_folder, file_prefix, window_size):
    # Get list of all CSV files in the input folder
    file_paths = glob.glob(f"{input_folder}/*.csv")
    
    if not file_paths:
        print(f"No CSV files found in {input_folder}")
        return
    
    # Compute mean and standard deviation across all files
    mean_std_df = compute_mean_and_std_across_files(file_paths)
    
    # Save mean and standard deviation DataFrame to CSV
    mean_output_path = f"{output_folder}/{file_prefix}_run_mean.csv"
    mean_std_df.to_csv(mean_output_path, index=False)
    print(f"Mean and standard deviation values saved to {mean_output_path}")
    
    # Compute rolling average and standard deviation from the mean DataFrame
    rolling_stats_df = compute_rolling_statistics(mean_std_df, window_size)
    
    # Save rolling statistics DataFrame to CSV
    rolling_avg_output_path = f"{output_folder}/{file_prefix}_rolling_average.csv"
    rolling_stats_df.to_csv(rolling_avg_output_path, index=False)
    print(f"Rolling average and standard deviation saved to {rolling_avg_output_path}")

# To run:  python compute_averages.py ./MZR/D-RND ./Averages drnd --window_size 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean, standard deviation, and rolling statistics of 'Value' column across multiple CSV files.")
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing CSV files')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save the results')
    parser.add_argument('file_prefix', type=str, help='Prefix to filenames')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for the rolling average (default: 100)')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.file_prefix, args.window_size)
