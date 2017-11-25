##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import time
import numpy as np
import json
from io import StringIO, BytesIO

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

import utils
from model import Net
from utils.mod_utils import Vgg16
from utils.img_utils import StyleLoader, InferenceStyleLoader, preprocess_batch

from options import Options
import logging
import traceback
import argparse

sagemaker_prefix = '/opt/ml'
input_path = os.path.join(sagemaker_prefix,'input/data')
output_path = os.path.join(sagemaker_prefix, 'output')
model_path = os.path.join(sagemaker_prefix, 'model')
param_path = os.path.join(sagemaker_prefix, 'input/config/hyperparameters.json')
temp_save_model_dir = os.path.join(sagemaker_prefix, 'pytorch-neural-art/models')
final_model_filename = "pytorch_neural_style_transfer.model"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SageMaker")

channel_name = 'train'
training_path = os.path.join(input_path, channel_name)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        with open(param_path,'r') as tc:
            
            trainingParams = json.load(tc)
            ngf = trainingParams.get('ngf', args.ngf)
            epochs = trainingParams.get('epochs', args.epochs)
            batch_size = trainingParams.get('batch_size',args.batch_size)
            log_interval = trainingParams.get('log_interval', args.log_interval)
            learning_rate = trainingParams.get('learning_rate',args.learning_rate)
            cuda = trainingParams.get('cuda', args.cuda)

            
            if cuda:
                logger.info("Using CUDA")
                torch.cuda.manual_seed(args.seed)
                kwargs = {'num_workers': 8, 'pin_memory': True}
                logger.info("Using kwarguments: \n" + str(kwargs))
            else:
                kwargs = {}
            
            transform = transforms.Compose([transforms.Scale(args.image_size),
                                            transforms.CenterCrop(args.image_size),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.mul(255))])
            train_dataset = datasets.ImageFolder(args.dataset, transform)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
            style_model = Net(ngf=ngf)

            print(style_model)

            optimizer = Adam(style_model.parameters(), learning_rate)
            mse_loss = torch.nn.MSELoss()

            vgg = Vgg16()

            utils.mod_utils.init_vgg16(args.vgg_model_dir)
            vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

            if cuda:
                style_model.cuda()
                vgg.cuda()
            
            style_loader = StyleLoader(args.style_folder, args.style_size)

            for e in range(epochs):
                style_model.train()
                agg_content_loss = 0.
                agg_style_loss = 0.
                count = 0

                for batch_id, (x, _) in enumerate(train_loader):
                    n_batch = len(x)
                    count += n_batch
                    optimizer.zero_grad()
                    x = Variable(preprocess_batch(x))
                    if cuda:
                        x.cuda()

                    style_v = style_loader.get(batch_id)
                    style_model.setTarget(style_v)

                    style_v = utils.img_utils.subtract_imagenet_mean_batch(style_v)
                    features_style = vgg(style_v)
                    gram_style = [utils.img_utils.gram_matrix(y) for y in features_style]

                    y = style_model(x.cuda())
                    xc = Variable(x.data.clone(), volatile=True)

                    y = utils.img_utils.subtract_imagenet_mean_batch(y)
                    xc = utils.img_utils.subtract_imagenet_mean_batch(xc)

                    features_y = vgg(y)
                    features_xc = vgg(xc.cuda())

                    f_xc_c = Variable(features_xc[1].data, requires_grad=False)

                    content_loss = args.content_weight * \
                        mse_loss(features_y[1], f_xc_c)

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

                    if (batch_id + 1) % log_interval == 0:
                        msg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                            time.ctime(), e + 1, count, len(train_dataset),
                            agg_content_loss / (batch_id + 1),
                            agg_style_loss / (batch_id + 1),
                            (agg_content_loss + agg_style_loss) / (batch_id + 1)

                        )
                        print(msg)
                    
                    if (batch_id + 1) % (20 * log_interval) == 0:
                        # save model
                        style_model.eval()
                        style_model.cpu()
                        save_model_filename = "Epoch_" + str(e) + "_" +\
                                              "iters_" + str(count) + \
                                              "_" + str(time.ctime()).replace(' ','_') + \
                                               "_" + str(args.content_weight) + "_" + \
                                               str(args.style_weight) + ".model"
                        save_model_path = os.path.join(temp_save_model_dir, save_model_filename)
                        
                        torch.save(style_model.state_dict(), save_model_path)
                        style_model.train()
                        style_model.cuda()
                        logger.info("Checkpoint, trained model saved at " + str(save_model_path))

            # save the final model

            style_model.eval()
            style_model.cpu()
            save_final_model_path = os.path.join(
                model_path, final_model_filename)
            torch.save(style_model.state_dict(), save_final_model_path)

            logger.info("Done, trained model saved at " + save_final_model_path)

            # Write out the success file
            with open(os.path.join(output_path, 'success'), 'w') as s:
                  s.write('Done')
         
    except Exception as e:
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            trc = traceback.format_exc()
            logger.info('Exception during training: ' +
                  str(e) + '\n' + trc)
            s.write('Exception during training: ' + str(e) + '\n' + trc)


def sigterm_handler(signum, frame):
    # Shutdown Flask application when SIGTERM is received as a result of "docker stop" command
    app.shutdown()


########################################################################################################
#
#                                        Hosting
#
########################################################################################################


# A singleton for holding the model
class ScoringService(object):
    PORT = 8080
    model = None

    @classmethod
    def get_model(cls):
        if not cls.is_model_loaded():
                # TODO: change the load method to the proper format.
                style_model = Net(ngf=args.ngf)
                cls.model = style_model.load_state_dict(torch.load(os.path.join(model_path, final_model_filename)))
        return cls.model

    @classmethod
    def is_model_loaded(cls):
        return cls.model is not None


    @classmethod
    def evaluate(cls, content_img, style_img):
        # basedir to save the data
        cls.model.eval()
        style_loader = InferenceStyleLoader(args.style_folder, style_img, args.style_size,
                                   cuda=args.cuda)

        content_image = utils.img_utils.tensor_load_inference_img(
                content_image, size=args.content_size, keep_asp=True).unsqueeze(0)
        if args.cuda:
            content_image = content_image.cuda()
            content_image = Variable(
                utils.img_utils.preprocess_batch(content_image), volatile=True)
            cls.model.cuda()

        style_v = Variable(style_loader.get().data, volatile=True)
        style_model.setTarget(style_v)
        output = style_model(content_image)

        # Generate the base64 inference image string       
        stylized_img = utils.img_utils.tensor_make_inference_img_str(
                output.data[0], args.cuda)
        return stylized_img



# The flask app for serving predictions
import signal
import flask
from flask import jsonify, request
import base64
import re
import json
from PIL import Image

app = flask.Flask(__name__)


@app.route("/ping", methods=["GET","POST"])
def ping():
    health = ScoringService.is_model_loaded()
    health = True

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    ScoringService.get_model()
    data = None

    # Convert from json to numpy
    if flask.request.content_type == "application/json":
        
        data = request.json()
        base64_user_photo = data['userPhoto']
        style_photo = data['stylePhoto']
        content_img = Image.open(BytesIO(base64.b64decode(base64_user_photo)))
        
        # Do the prediction. Get the stylizedPhoto in base64
        stylizedPhoto = ScoringService.evaluate(content_img, style_photo)

        # Results

        result = jsonify(
            {
                "stylizedPhoto": stylizedPhoto
            }
        )
        return flask.Response(response=result, status=200, mimetype="application/json")

    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype="text/plain")



def serve():
    signal.signal(signal.SIGTERM, sigterm_handler)
    app.run(host="0.0.0.0", port=ScoringService.PORT)


def main():

# The main routine decides what mode we're in and executes that arm
    main_args = Options()
    args = main_args.parser.parse_args()
    if args.subcommand == "train":
        logger.info("Training with arguments: " + str(args))
        train(args)
    else:
        eval_args = {
            "style_folder": "images/9styles/",
            "save_model_dir": "/opt/ml/model/",
            "image_size": 256,
            "style_size": 512,
            "cuda": 1
        }
        main_args.parser.set_defaults(**eval_args)
        args = main_args.parser.parse_args()
        logger.info("Serving with arguments: " + str(args))
        serve()


if __name__ == "__main__":
    main()
    