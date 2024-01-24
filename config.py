# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
weight_decay = 1e-5
init_scale = 0.01

# Model:
device_ids = [0]

# Train:
batch_size = 8
cropsize = 64
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.5
num_train = 1000

# Val:
cropsize_val = 64
batchsize_val = 2
shuffle_val = False
val_freq = 10
num_val = 200


# model
load_path = "drive/MyDrive/PRIS/models/model_checkpoint_00750.pt"
start_epoch = 750
end_epoch = 1600
step = 0 # 0: pretrain, 1: enhance, 2: finetune


# Dataset
dataset = "imagenet"
TRAIN_PATH = 'data/ImageNet/train'
VAL_PATH = 'data/ImageNet/test'
format_train = 'JPEG'
format_val = 'JPEG'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['train loss', 'val loss', 'lr', 'attack method']
silent = False
live_visualization = False
progress_bar = True


# Saving checkpoints:

MODEL_PATH = 'drive/MyDrive/PRIS/models'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = 'drive/MyDrive/PRIS/image/'
IMAGE_PATH_host= IMAGE_PATH + 'host/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_container = IMAGE_PATH + 'container/'
IMAGE_PATH_extracted = IMAGE_PATH + 'extracted/'

