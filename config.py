# Configuring the DCGAN model.
class Config:
    def __init__(self):
        # Root directory for the dataset.
        self.dataroot = "/Users/simpleai/Desktop/DCGAN/dataset/celeba"
        # Number of workers for the dataloader.
        self.workers = 2
        # Batch size during training.
        self.batch_size = 512
        # Spatial size of training images. All images will be resized to this
        #    size using a transformer.
        self.image_size = 64
        # Number of channels in the images. For Color images, the number of channels is 3.
        self.nc = 3
        # Size of the latent z vector. (size of the generator input)
        self.nz = 100
        # Size of features in the generator.
        self.ngf = 64
        # Size of features in the discriminator.
        self.ndf = 64
        # Number of epochs to train for.
        self.num_epochs = 50
        # Learning rate for the optimizer.
        self.lr = 0.0002
        # Beta1 hyperparameter for the Adam optimizer.
        self.beta1 = 0.5
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = 1


