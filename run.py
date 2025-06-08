import subprocess

if __name__ == '__main__':
    dataset = 'vsumme'  # Change to 'summe' if needed
    for i in range(1, 6):
        print(f"Running fold {i}...")
        cmd = [
            'python', 'train.py',
            '-e', 
            '-i', f'input_folder/{dataset}/fold_{i}/train',
            '-o', f'features_folder/{dataset}/fold_{i}_train.h5',
            '--save_model', f'weights/{dataset}/fold_{i}.pth'
        ]
        subprocess.run(cmd, check=True)
        cmd = [
            'python', 'test.py',
            '-e',
            '-i', f'input_folder/{dataset}/fold_{i}/val',
            '-ft', f'features_folder/{dataset}/fold_{i}_val.h5',
            '--pretrained_model', f'weights/{dataset}/fold_{i}.pth',
            '-o', f'summaries/{dataset}/fold_{i}_summary.h5',
            '-v', f'visualize_folder/{dataset}/fold_{i}',
            '--user_summary', f'user_summaries/{dataset}',
            '--save_results', f'results/{dataset}/fold_{i}.txt'
        ]
        subprocess.run(cmd, check=True)
        print(f"Fold {i} completed.\n")