import torch
import torch.nn
import torch.optim
import torchvision
from model import *
import config as c
import datasets
import modules.Unet_common as common
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_residual(img1, img2, scale): return torch.clamp(torch.abs(img1 - img2) * scale, 0.0, 1.0)

def load(name, net, optim):
    state_dicts = torch.load(name, weights_only=False)
    network_state_dict = {
        k: v
        for k, v in state_dicts['net'].items()
        if 'tmp_var' not in k
    }
    net.load_state_dict(network_state_dict)

    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def split_three(data):
    if data.shape[0] % 3 != 0:
        raise ValueError(
            'The batch size must be divisible by 3, '
            f'but got {data.shape[0]}.'
        )

    group_size = data.shape[0] // 3

    cover = data[:group_size, :, :, :]
    secret_1 = data[group_size:2 * group_size, :, :, :]
    secret_2 = data[2 * group_size:3 * group_size, :, :, :]

    return cover, secret_1, secret_2


if __name__ == '__main__':
    print('Start testing...')
    residual_scale = getattr(c, 'RESIDUAL_SCALE', 20)

    for path in [
        c.IMAGE_PATH_cover,
        c.IMAGE_PATH_secret_1,
        c.IMAGE_PATH_secret_2,
        c.IMAGE_PATH_steg_1,
        c.IMAGE_PATH_steg_2,
        c.IMAGE_PATH_secret_rev_1,
        c.IMAGE_PATH_secret_rev_2,
        c.IMAGE_PATH_resi_cover,
        c.IMAGE2_PATH_resi_secret_1,
        c.IMAGE2_PATH_resi_secret_2,
    ]:
        os.makedirs(path, exist_ok=True)

    net1 = Model_1()
    net2 = Model_2()

    net1.cuda()
    net2.cuda()

    init_model(net1)
    init_model(net2)

    net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
    net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)

    params_trainable_1 = list(
        filter(lambda p: p.requires_grad, net1.parameters())
    )
    params_trainable_2 = list(
        filter(lambda p: p.requires_grad, net2.parameters())
    )

    optim1 = torch.optim.Adam(params_trainable_1,lr=c.lr,betas=c.betas, eps=1e-6, weight_decay=c.weight_decay )
    optim2 = torch.optim.Adam(     params_trainable_2,    lr=c.lr,    betas=c.betas,     eps=1e-6,     weight_decay=c.weight_decay )

    load(    os.path.join(c.MODEL_PATH_2, 'model_best_1.pt'),     net1,     optim1 )
    load( os.path.join(c.MODEL_PATH_2, 'model_best_2.pt'), net2,   optim2 )

    net1.eval()
    net2.eval()

    dwt = common.DWT()
    iwt = common.IWT()

    with torch.no_grad():
        for i, data in enumerate(datasets.testloader):
            data = data.to(device)

            cover, secret_1, secret_2 = split_three(data)

            cover_input = dwt(cover)
            secret_input_1 = dwt(secret_1)
            secret_input_2 = dwt(secret_2)

            input_img_1 = torch.cat(      (cover_input, secret_input_1),   1     )
            output_1, _ = net1(input_img_1)
            output_steg_1 = output_1.narrow(       1,       0,      4 * c.channels_in    )
            steg_img_1 = iwt(output_steg_1)
            input_img_2 = torch.cat(    (output_steg_1, secret_input_2),     1 )

            output_2, _ = net2(input_img_2)
            output_steg_2 = output_2.narrow(  1,    0,4 * c.channels_in)
            steg_img_2 = iwt(output_steg_2)
            backward_img_2 = net2(output_steg_2,  rev=True)

            steg_rev_1 = backward_img_2.narrow( 1,0,4 * c.channels_in)
            secret_rev_2 = backward_img_2.narrow(1,4 * c.channels_in,backward_img_2.shape[1] - 4 * c.channels_in )
            secret_rev_2 = iwt(secret_rev_2)
            backward_img_1 = net1( steg_rev_1, rev=True)

            cover_rev = backward_img_1.narrow(1, 0, 4 * c.channels_in)
            cover_rev = iwt(cover_rev)
            secret_rev_1 = backward_img_1.narrow( 1, 4 * c.channels_in, backward_img_1.shape[1] - 4 * c.channels_in)
            secret_rev_1 = iwt(secret_rev_1)

            resi_cover = make_residual(steg_img_2, cover, residual_scale)
            resi_secret_1 = make_residual(secret_rev_1, secret_1, residual_scale)
            resi_secret_2 = make_residual(secret_rev_2, secret_2, residual_scale)

            torchvision.utils.save_image(cover,c.IMAGE_PATH_cover + '%.5d.png' % i)
            torchvision.utils.save_image(secret_1,c.IMAGE_PATH_secret_1 + '%.5d.png' % i)
            torchvision.utils.save_image(secret_2, c.IMAGE_PATH_secret_2 + '%.5d.png' % i )
            torchvision.utils.save_image(steg_img_1,c.IMAGE_PATH_steg_1 + '%.5d.png' % i )
            torchvision.utils.save_image(steg_img_2,c.IMAGE_PATH_steg_2 + '%.5d.png' % i)
            torchvision.utils.save_image(secret_rev_1,c.IMAGE_PATH_secret_rev_1 + '%.5d.png' % i)
            torchvision.utils.save_image(secret_rev_2,c.IMAGE_PATH_secret_rev_2 + '%.5d.png' % i)

            torchvision.utils.save_image(resi_cover, c.IMAGE2_PATH_resi_cover + '%.5d.png' % i)
            torchvision.utils.save_image(resi_secret_1, c.IMAGE2_PATH_resi_secret_1 + '%.5d.png' % i)
            torchvision.utils.save_image(resi_secret_2, c.IMAGE2_PATH_resi_secret_2 + '%.5d.png' % i)
            print(f'Saved residuals for image {i}')

    print('Finished testing.')