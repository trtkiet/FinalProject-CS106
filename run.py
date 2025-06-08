import subprocess

if __name__ == '__main__':
    for i in range(1, 6):
        print(f"Running fold {i}...")
        cmd = [
            'python', 'train.py',
            '-e', 
            '-i', f'input_folder/vsumme/fold_{i}/train',
            '-o', f'features_folder/vsumme/fold_{i}_train.h5',
            '--save_model', f'weights/vsumme/fold_{i}.pth'
        ]
        subprocess.run(cmd, check=True)
        cmd = [
            'python', 'test.py',
            '-e',
            '-i', f'input_folder/vsumme/fold_{i}/val',
            '-ft', f'features_folder/vsumme/fold_{i}_val.h5',
            '--pretrained_model', f'weights/vsumme/fold_{i}.pth',
            '-o', f'summaries/vsumme/fold_{i}_summary.h5',
            '-v', f'visualize_folder/vsumme/fold_{i}'
        ]
        subprocess.run(cmd, check=True)
        print(f"Fold {i} completed.\n")