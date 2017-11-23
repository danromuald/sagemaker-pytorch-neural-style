import argparse


class TrainingOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for training PyTorch-Style-Transfer")
        # training args
        self.parser.add_argument("--ngf", type=int, default=128,
                                help="number of generator filter channels, default 128")
        self.parser.add_argument("--epochs", type=int, default=2,
                                help="number of training epochs, default is 2")
        self.parser.add_argument("--batch-size", type=int, default=4,
                                help="batch size for training, default is 4")
        self.parser.add_argument("--dataset", type=str, default="/opt/ml/input/data/",
                                help="path to training dataset, the path should point to a folder "
                                "containing another folder with all the training images")
        self.parser.add_argument("--style-folder", type=str, default="images/9styles/",
                                help="path to style-folder")
        self.parser.add_argument("--vgg-model-dir", type=str, default="models/",
                                help="directory for vgg, if model is not present in the directory it is downloaded")
        self.parser.add_argument("--save-model-dir", type=str, default="/opt/ml/model/",
                                help="path to folder where trained model will be saved.")
        self.parser.add_argument("--image-size", type=int, default=256,
                                help="size of training images, default is 256 X 256")
        self.parser.add_argument("--style-size", type=int, default=512,
                                help="size of style-image, default is the original size of style image")
        self.parser.add_argument("--cuda", type=int, default=1,
                                help="set it to 1 for running on GPU, 0 for CPU")
        self.parser.add_argument("--seed", type=int, default=42,
                                help="random seed for training")
        self.parser.add_argument("--content-weight", type=float, default=1.0,
                                help="weight for content-loss, default is 1.0")
        self.parser.add_argument("--style-weight", type=float, default=5.0,
                                help="weight for style-loss, default is 5.0")
        self.parser.add_argument("--learning-rate", type=float, default=1e-3,
                                help="learning rate, default is 0.001")
        self.parser.add_argument("--log-interval", type=int, default=500,
                                help="number of images after which the training loss is logged, default is 500")
        self.parser.add_argument("--resume", type=str, default=None,
                                help="resume if needed")

        #return self.parser.parse_args()


class EvalOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="parser for evaluating PyTorch-Style-Transfer")

        # evaluation args
        self.parser.add_argument("--ngf", type=int, default=128,
                                help="number of generator filter channels, default 128")
        self.parser.add_argument("--content-image", type=str, required=False,
                                help="path to content image you want to stylize")
        self.parser.add_argument("--style-image", type=str, default="images/9styles/candy.jpg",
                                help="path to style-image")
        self.parser.add_argument("--content-size", type=int, default=512,
                                help="factor for scaling down the content image")
        self.parser.add_argument("--style-size", type=int, default=512,
                                help="size of style-image, default is the original size of style image")
        self.parser.add_argument("--style-folder", type=str, default="images/9styles/",
                                help="path to style-folder")
        self.parser.add_argument("--output-image", type=str, default="output.jpg",
                                help="path for saving the output image")
        self.parser.add_argument("--model", type=str, required=False,
                                help="saved model to be used for stylizing the image")
        self.parser.add_argument("--cuda", type=int, default=1,
                                help="set it to 1 for running on GPU, 0 for CPU")
        self.parser.add_argument("--vgg-model-dir", type=str, default="models/",
                                help="directory for vgg, if model is not present in the directory it is downloaded")