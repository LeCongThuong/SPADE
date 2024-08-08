# (epoch: 1, iters: 52, time: 0.259) KLD: 0.349 GAN: -0.082 GAN_Feat: 0.393 VGG: 0.254 MSE: 0.472 D_Fake: 1.082 D_real: 1.008 


import re
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


# Define the regex pattern to capture key-value pairs in the text
pattern = r'\(epoch: (\d+), iters: (\d+), time: ([\d.]+)\) KLD: ([\d.-]+) GAN: ([\d.-]+) GAN_Feat: ([\d.-]+) VGG: ([\d.-]+) MSE: ([\d.-]+) D_Fake: ([\d.-]+) D_real: ([\d.-]+)'

# Initialize an empty list to store the results
results = []

with open('/mnt/hmi/thuong/SPADE/checkpoints/multi_spade_1/loss_log.txt', 'r') as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
                # Create a dictionary with the extracted values
                data = {
                    'epoch': int(match.group(1)),
                    'iters': int(match.group(2)),
                    'time': float(match.group(3)),
                    'KLD': float(match.group(4)),
                    'GAN': float(match.group(5)),
                    'GAN_Feat': float(match.group(6)),
                    'VGG': float(match.group(7)),
                    'MSE': float(match.group(8)),
                    'D_Fake': float(match.group(9)),
                    'D_real': float(match.group(10))
                }
                # Add the dictionary to the results list
                results.append(data)


df = pd.DataFrame(results)
df[['epoch', 'iters']] = df[['epoch', 'iters']].astype(int)
max_iters = df['iters'].max()  # Assuming max iters is known or calculated from data

avg_metrics_per_epoch = df.groupby('epoch').mean()

# Create a SummaryWriter instance
writer = SummaryWriter('experiment_metrics')


for index, row in df.iterrows():
    step = (row['epoch'] - 1) * max_iters + row['iters']
    metrics = ['KLD', 'GAN', 'GAN_Feat', 'VGG', 'MSE', 'D_Fake', 'D_real']
    for metric in metrics:
        writer.add_scalar(tag=metric, scalar_value=float(row[metric]), global_step=float(step))
    writer.flush()

for epoch, row in avg_metrics_per_epoch.iterrows():
    for metric in ['KLD', 'GAN', 'GAN_Feat', 'VGG', 'MSE', 'D_Fake', 'D_real']:
        writer.add_scalar(tag=f'epoch_{metric}', scalar_value=row[metric], global_step=epoch)
    writer.flush()

writer.close()
print("Data has been logged to TensorBoard.")

