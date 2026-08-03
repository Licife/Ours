# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 2
lamda_guide = 1
lamda_low_frequency = 1
lamda_alm = 1

device_ids = [0]

# Train:
batch_size = 12
cropsize = 128
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.5

# Val:
cropsize_val = 1024
# cropsize_val = 256 # DIV2K use 1024, ImageNet and COCO use 256
batchsize_val = 3
shuffle_val = False
val_freq = 5000
save_freq = 10


TRAIN_PATH = r'D:/Datasets/DIV2K/DIV2K_train_HR'
format_train = 'png'

# Load:
suffix = 'model_best.pt'
tain_next = False
trained_epoch = 0

IMAGE_PATH_1 = 'image1/'
IMAGE_PATH_2 = 'image2/'

residual = True
residual_scale = 20
IMAGE_PATH_resi_cover = IMAGE_PATH_1 + 'resi_c/'
IMAGE_PATH_resi_secret = IMAGE_PATH_1 + 'resi_s/'


IMAGE2_PATH_resi_cover = IMAGE_PATH_2 + 'resi_c/'
IMAGE2_PATH_resi_secret_1 = IMAGE_PATH_2 + 'resi_s1/'
IMAGE2_PATH_resi_secret_2 = IMAGE_PATH_2 + 'resi_s2/'

# Dataset DIV2K
VAL_PATH = r'D:/Datasets/DIV2K/DIV2K_valid_HR'
format_val = 'png'
# IMAGE_PATH_cover = IMAGE_PATH_1 + 'cover_d/'
# IMAGE_PATH_secret = IMAGE_PATH_1 + 'secret_d/'
# IMAGE_PATH_steg = IMAGE_PATH_1 + 'steg_d/'
# IMAGE_PATH_secret_rev = IMAGE_PATH_1 + 'secret-rev_d/'

IMAGE_PATH_cover = IMAGE_PATH_2 + 'cover_d/'
IMAGE_PATH_secret_1 = IMAGE_PATH_2 + 'secret_1_d/'
IMAGE_PATH_secret_2 = IMAGE_PATH_2 + 'secret_2_d/'
IMAGE_PATH_steg_1 = IMAGE_PATH_2 + 'steg_1_d/'
IMAGE_PATH_steg_2 = IMAGE_PATH_2 + 'steg_2_d/'
IMAGE_PATH_secret_rev_1 = IMAGE_PATH_2 + 'secret-rev_1_d/'
IMAGE_PATH_secret_rev_2 = IMAGE_PATH_2 + 'secret-rev_2_d/'

# Dataset COCO
# VAL_PATH = r'D:/Datasets/COCO2017/val2017'
# format_val = 'jpg'
# IMAGE_PATH_cover = IMAGE_PATH_1 + 'cover_c/'
# IMAGE_PATH_secret = IMAGE_PATH_1 + 'secret_c/'
# IMAGE_PATH_steg = IMAGE_PATH_1 + 'steg_c/'
# IMAGE_PATH_secret_rev = IMAGE_PATH_1 + 'secret-rev_c/'

# IMAGE_PATH_cover = IMAGE_PATH_2 + 'cover_c/'
# IMAGE_PATH_secret_1 = IMAGE_PATH_2 + 'secret_1_c/'
# IMAGE_PATH_secret_2 = IMAGE_PATH_2 + 'secret_2_c/'
# IMAGE_PATH_steg_1 = IMAGE_PATH_2 + 'steg_1_c/'
# IMAGE_PATH_steg_2 = IMAGE_PATH_2 + 'steg_2_c/'
# IMAGE_PATH_secret_rev_1 = IMAGE_PATH_2 + 'secret-rev_1_c/'
# IMAGE_PATH_secret_rev_2 = IMAGE_PATH_2 + 'secret-rev_2_c/'

# Dataset ImageNet
# VAL_PATH = r'D:/Datasets/ImageNET/train'
# format_val = 'JPEG'
# IMAGE_PATH_cover = IMAGE_PATH_1 + 'cover_i/'
# IMAGE_PATH_secret = IMAGE_PATH_1 + 'secret_i/'
# IMAGE_PATH_steg = IMAGE_PATH_1 + 'steg_i/'
# IMAGE_PATH_secret_rev = IMAGE_PATH_1 + 'secret-rev_i/'

# IMAGE_PATH_cover = IMAGE_PATH_2 + 'cover_i/'
# IMAGE_PATH_secret_1 = IMAGE_PATH_2 + 'secret_1_i/'
# IMAGE_PATH_secret_2 = IMAGE_PATH_2 + 'secret_2_i/'
# IMAGE_PATH_steg_1 = IMAGE_PATH_2 + 'steg_1_i/'
# IMAGE_PATH_steg_2 = IMAGE_PATH_2 + 'steg_2_i/'
# IMAGE_PATH_secret_rev_1 = IMAGE_PATH_2 + 'secret-rev_1_i/'
# IMAGE_PATH_secret_rev_2 = IMAGE_PATH_2 + 'secret-rev_2_i/'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:
LOG_PATH = 'log/'

MODEL_PATH_1 = 'model1/'
MODEL_PATH_2 = 'model2/'
MODEL_PATH_3 = 'model3/'
MODEL_PATH_4 = 'model4/'
MODEL_PATH_5 = 'model5/'
checkpoint_on_error = True
SAVE_freq = 50


