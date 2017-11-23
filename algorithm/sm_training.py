import os
import sys
import time
import numpy as np

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
from model import Net
from utils.mod_utils import Vgg16
from utils.img_utils import StyleLoader


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


class SMTraining(object):
    def __init__(self, *args):
        super(SMTraining, self).__init__(*args))
        self.args = args

    @classmethod
    def train(self, args):
        check_paths(args)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            kwargs = {'num_workers': 4, 'pin_memory': True}
        else:
            kwargs = {}

        transform = transforms.Compose([transforms.Scale(args.image_size),
                                        transforms.CenterCrop(args.image_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.mul(255))])
        train_dataset = datasets.ImageFolder(args.dataset, transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

        style_model = Net(ngf=args.ngf)

        if args.resume is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume))
            style_model.load_state_dict(torch.load(args.resume))
        
        print(style_model)
        optimizer = Adam(style_model.parameters(), args.lr)
        mse_loss = torch.nn.MSELoss()

        vgg = Vgg16()

        utils.mod_utils.init_vgg16(args.vgg_model_dir)
        vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

        if args.cuda:
            style_model.cuda()
            vgg.cuda()

        style_loader = utils.StyleLoader(args.style_folder, args.style_size)

        for e in range(args.epochs):
            style_model.train()
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()
                x = Variable(utils.img_utils.preprocess_batch(x))
                if args.cuda:
                    x = x.cuda()

                style_v = style_loader.get(batch_id)
                style_model.setTarget(style_v)

                style_v = utils.img_utils.subtract_imagenet_mean_batch(style_v)
                features_style = vgg(style_v)
                gram_style = [utils.img_utils.gram_matrix(y) for y in features_style]

                y = style_model(x)
                xc = Variable(x.data.clone(), volatile=True)

                y = utils.img_utils.subtract_imagenet_mean_batch(y)
                xc = utils.img_utils.subtract_imagenet_mean_batch(xc)

                features_y = vgg(y)
                features_xc = vgg(xc)

                f_xc_c = Variable(features_xc[1].data, requires_grad=False)

                content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

                style_loss = 0.
                for m in range(len(features_y)):
                    gram_y = utils.img_utils.gram_matrix(features_y[m])
                    gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(args.batch_size, 1, 1, 1)
                    style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.data[0]
                agg_style_loss += style_loss.data[0]

                if (batch_id + 1) % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    print(mesg)

                if (batch_id + 1) % (4 * args.log_interval) == 0:
                    # save model
                    style_model.eval()
                    style_model.cpu()
                    save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + str(time.ctime()).replace(' ',
                                                                                                                    '_') + "_" + str(
                        args.content_weight) + "_" + str(args.style_weight) + ".model"
                    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                    torch.save(style_model.state_dict(), save_model_path)
                    style_model.train()
                    style_model.cuda()
                    print("\nCheckpoint, trained model saved at", save_model_path)

        # save model
        style_model.eval()
        style_model.cpu()
        save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
            args.content_weight) + "_" + str(args.style_weight) + ".model"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(style_model.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)

    ########### EVALUATION ##########

    @classmethod
    def evaluate(args):
        content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
        content_image = content_image.unsqueeze(0)

        style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
        style = style.unsqueeze(0)
        style = utils.preprocess_batch(style)

        style_model = Net(ngf=args.ngf)
        style_model.load_state_dict(torch.load(args.model))

        if args.cuda:
            style_model.cuda()
            content_image = content_image.cuda()
            style = style.cuda()

        style_v = Variable(style, volatile=True)

        content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
        style_model.setTarget(style_v)

        output = style_model(content_image)

        utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)
        
        