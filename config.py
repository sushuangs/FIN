epochs = 500
clamp = 2.0

# optimizer
lr = 1e-4
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-5

noise_flag = True

# input settings
message_weight = 10
stego_weight = 1
message_length = 30

# Train:
batch_size = 32
cropsize = 128

# Val:
batchsize_val = 32
cropsize_val = 128

# Data Path
TRAIN_PATH = '/data/experiment/data/gtos128_500/train'
VAL_PATH = '/data/experiment/data/gtos128_500/val'

format_train = 'png'
format_val = 'png'

# Saving checkpoints:
MODEL_PATH = 'experiments/gtos_I_GN/'
SAVE_freq = 1

suffix = 'fed_19.149027_00004.pt'
train_continue = True
diff = True






