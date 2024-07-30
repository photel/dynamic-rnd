import csv
import argparse
import math

def compute_average_of_last_100_values(csv_file_path, stop_step):
    '''
        Compute average of Value column over the previous 100 rows when Step exceeds a given value.
        Note that 1 step = 4 frames
    '''
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        last_100_values = []
        
        for row in reader:
            step = int(row['Step'])
            value = float(row['Value'])
            
            last_100_values.append(value)
            
            if len(last_100_values) > 100:
                last_100_values.pop(0)
            
            if step > stop_step:
                average_value = sum(last_100_values) / len(last_100_values)

                variance = sum((x - average_value) ** 2 for x in last_100_values) / len(last_100_values)
                std_deviation = math.sqrt(variance)
                
                return average_value, std_deviation
            
                return average_value
        
        return None


# To run: python get_staged_scores.py filename.csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average of Value column over the previous 100 rows when Step exceeds a given value.')
    parser.add_argument('csv_file_path', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    
    results = {
        'average_value': [],
        'standard_dev': []
    }

    if 'Breakout' in args.csv_file_path:
        steps = [2_500_000, 9_988_000]
    else:
        steps = [2_500_000, 12_500_000, 25_000_000, 49_990_000]

    for s in steps:
        avg, std = compute_average_of_last_100_values(args.csv_file_path, s)
        results['average_value'].append(avg)
        results['standard_dev'].append(std)
    if results['average_value'] is not None:
        print(f'The average returns over the 100 episodes at the specified stop steps are: {results["average_value"]}')
        print(f'Standard deviations are: {results["standard_dev"]}')
    else:
        print('No step exceeded the step value in the provided CSV file.')
