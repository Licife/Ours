# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.0
lr = 10 ** log10_lr
warmup_epochs = 10
epochs = 1000 + warmup_epochs
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 2
lamda_guide = 1
lamda_low_frequency = 1
lamda_alm = 1

device_ids = [0]

# Train:
batch_size = 4
cropsize = 256
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.5

# Val:
cropsize_val = 1024
# cropsize_val = 256 # DIV2K use 1024, ImageNet and COCO use 256
batchsize_val = 2
shuffle_val = False
val_freq = 50
save_freq = 10


TRAIN_PATH = r'D:/Datasets/DIV2K/DIV2K_train_HR'
format_train = 'png'

# Load:
suffix = 'model_best.pt'
tain_next = False
trained_epoch = 0

# Dataset DIV2K
VAL_PATH = r'D:/Datasets/DIV2K/DIV2K_valid_HR'
format_val = 'png'
IMAGE_PATH = 'image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover_d/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret_d/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg_d/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev_d/'
IMAGE_PATH_resi_cover = IMAGE_PATH + 'resi_c_d/'
IMAGE_PATH_resi_secret = IMAGE_PATH + 'resi_s_d/'

# Dataset COCO
# VAL_PATH = r'D:/Datasets/COCO2017/val2017'
# format_val = 'jpg'
# IMAGE_PATH = 'image/'
# IMAGE_PATH_cover = IMAGE_PATH + 'cover_c/'
# IMAGE_PATH_secret = IMAGE_PATH + 'secret_c/'
# IMAGE_PATH_steg = IMAGE_PATH + 'steg_c/'
# IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev_c/'
# IMAGE_PATH_resi_cover = IMAGE_PATH + 'resi_c_c/'
# IMAGE_PATH_resi_secret = IMAGE_PATH + 'resi_s_c/'

# Dataset ImageNet
# VAL_PATH = r'D:/Datasets/ImageNET/train'
# format_val = 'JPEG'
# IMAGE_PATH = 'image/'
# IMAGE_PATH_cover = IMAGE_PATH + 'cover_i/'
# IMAGE_PATH_secret = IMAGE_PATH + 'secret_i/'
# IMAGE_PATH_steg = IMAGE_PATH + 'steg_i/'
# IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev_i/'
# IMAGE_PATH_resi_cover = IMAGE_PATH + 'resi_c_i/'
# IMAGE_PATH_resi_secret = IMAGE_PATH + 'resi_s_i/'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:
LOG_PATH = 'log/'

MODEL_PATH = 'model/'
checkpoint_on_error = True
SAVE_freq = 50


