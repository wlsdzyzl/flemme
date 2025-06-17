import os
### by organ to by number
dataset_path = "/media/wlsdzyzl/DATA1/datasets/pcd/MedPointS/"
selected_numbers = []
with open(os.path.join(dataset_path, 'MedPointSDataset.txt'), 'r') as f:
    lines = f.readlines()
for l in lines:
    if 'vertebrae_T9' in l or 'vertebraeT9' in l:
        selected_numbers.append(l.split('_')[0].split('=')[-1])

for sid, s in enumerate(selected_numbers):
    print(f'extract organs of the {sid}th person')
    os.mkdir(f'{dataset_path}/by_number/{s}')
    command = f'cp {dataset_path}/by_organ/*/*{s}*.stl {dataset_path}/by_number/{s}/ --verbose'
    os.system(command)
