import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

class Unconditional(object):
    def __init__(self, dimension, generator, latent_size, latent_dist, use_cuda=False):
        self.use_cuda = use_cuda

        self.dimension = dimension
        self.generator = generator

        if self.use_cuda:
            self.generator = self.generator.cuda()

        self.latent_size = latent_size
        self.latent_dist = latent_dist

    def sample_batch(self, batch, m):
        config = [batch, self.latent_size] + m
        zhat = None
        if self.use_cuda:
            zhat = Variable(torch.FloatTensor(*config).cuda(), requires_grad=True)
        else:
            zhat = Variable(torch.FloatTensor(*config), requires_grad=True)
        zhat.retain_grad()
        if self.latent_dist == "normal":
            zhat.data = zhat.data.normal_(0, 1)
        elif self.latent_dist == "uniform":
            zhat.data = zhat.data.uniform_(-1, 1)
        return zhat

    def create_unconditional_simulations(self, batch, m):
        zhat = self.sample_batch(batch, m)

        if self.use_cuda:
            zhat = zhat.cuda()
        try:
            xhat = self.generator(zhat)
            return xhat.data.cpu().numpy()
        except RuntimeError:
            print("Out of GPU Memory.")


class Conditioner(object):
    def __init__(self, dimension, generator, latent_size, conditioning_data, mask, real_sample=None, latent_dist="normal", discriminator=None, use_cuda=False, verbose=False, tensorboard=None, is_binary=False, conditioner_vec=None):
        self.use_cuda = use_cuda

        self.dimension = dimension
        self.generator = generator
        self.latent_size = latent_size
        self.latent_dist = latent_dist

        self.discriminator = discriminator

        if self.discriminator is not None:
            for p in self.discriminator.parameters():
                p.requires_grad = False

        for p in self.generator.parameters():
            p.requires_grad = False

        if self.use_cuda:
            self.generator = self.generator.cuda()
            if self.discriminator is not None:
                self.discriminator = self.discriminator.cuda()

        self.conditioning_data = conditioning_data
        self.mask = mask
        self.unique = np.unique(self.mask, return_counts=True)[1][1]
        self.optimizer = None
        self.count = [0]
        self.verbose = verbose
        self.tensorboard = tensorboard
        self.is_binary = is_binary
        self.conditioner_vec = conditioner_vec
        self.real_sample = real_sample
        self.first = True
        self.content_loss = None
        self.perceptual = None
        self.total_loss = None
        self.content_criterion = nn.MSELoss(size_average=False)

    def sample_batch(self, batch, m):
        config = [batch, self.latent_size]+m
        if self.use_cuda:
            zhat = Variable(torch.FloatTensor(*config).cuda(), requires_grad=True)
        else:
            zhat = Variable(torch.FloatTensor(*config), requires_grad=True)
        zhat.retain_grad()
        if self.latent_dist == "normal":
            zhat.data = zhat.data.normal_(0, 1)
        elif self.latent_dist == "uniform":
            zhat.data = zhat.data.uniform_(-1, 1)
        return zhat

    def condition(self, batch, m, target):
        self.batch = batch
        self.m = m
        self.target = target
        self.zhat = self.sample_batch(batch, m)
        self.optimizer = optim.LBFGS([self.zhat])
        self.perceptual_losses, self.content_losses, self.total_losses = [999999], [999999], [999999]

        self.conditioning_data = Variable(torch.from_numpy(np.array([np.expand_dims(self.conditioning_data, 0)]*self.batch)).float())
        self.mask = Variable(torch.from_numpy(np.array([np.expand_dims(self.mask, 0)]*self.batch)).float())

        if self.use_cuda:
            self.conditioning_data = self.conditioning_data.cuda()
            self.mask = self.mask.cuda()

        self.conditioning_masked = self.conditioning_data * self.mask
        self.steps = 0
        while self.content_losses[-1] > target:
            self.optimizer.step(self.closure)
        print("Step: ", self.steps, " current loss: ", self.content_losses[-1], " target value: ", target)
        self.steps += 1

        return None

    def condition_logloss(self, batch, m, steps):
        self.batch = batch
        self.m = m
        self.zhat = self.sample_batch(batch, m)
        self.optimizer = optim.Adam([self.zhat], lr=1e-1, betas=(0.5, 0.9))
        self.perceptual_losses, self.content_losses, self.total_losses = [], [], []
        self.real_sample = Variable(torch.from_numpy(np.array([np.expand_dims(self.real_sample, 0)]*self.batch)).float())
        self.conditioning_data = Variable(torch.from_numpy(np.array([np.expand_dims(self.conditioning_data, 0)]*self.batch)).float())
        self.mask = Variable(torch.from_numpy(np.array([np.expand_dims(self.mask, 0)]*self.batch)).float())
        if self.use_cuda:
            self.conditioning_data = self.conditioning_data.cuda()
            self.mask = self.mask.cuda()
        self.real_sample = self.real_sample.cuda()

        disc_real = None
        if self.discriminator:
            disc_real = self.discriminator(self.real_sample).mean()

        self.criterion = nn.BCELoss(weight=self.mask, size_average=False)
        for i in range(steps):
            self.optimizer.zero_grad()
            completed = self.generator(self.zhat)
            content_loss = self.criterion(completed*0.5+0.5, self.conditioning_data)
            perceptual = None

            if self.discriminator:
                perceptual = self.discriminator(completed).mean()
                perceptual = (disc_real - perceptual).abs()
                perceptual.backward(retain_graph=True)

            content_loss.backward(retain_graph=True)
            total_loss = content_loss
            if self.discriminator:
                total_loss = content_loss - perceptual

            out = np.where((completed*0.5+0.5).data.cpu().numpy()[0, 0][:, 64, 64].reshape(-1) >= 0.5, 1., 0)
            bin_labels = self.conditioning_data.data.cpu().numpy()[0, 0][:, 64, 64].reshape(-1)

            if self.verbose:
                print("Iteration: ", i, " Current accuracy: ", accuracy_score(out, bin_labels))

            percep_np, content_np, total_np = self.get_numpy_values([perceptual, content_loss, total_loss])
            self.append_losses([percep_np, content_np, total_np])

            if f1_score(out, bin_labels) == 1.0 and percep_np < 0.1:
                print("Finished conditioning in ", i, " steps.")
                break

            self.optimizer.step()

            if i == 50:
                print("reducing lr")
                self.optimizer = optim.Adam([self.zhat], lr=1e-4, betas=(0.5, 0.9))

        return i, percep_np, content_np, total_np, self.zhat

    def closure(self):
        self.optimizer.zero_grad()
        completed = self.generator(self.zhat)

        if self.is_binary:
            completed.data = torch.sign(completed.data)

        completed_masked = completed*self.mask

        self.content_loss = torch.sum(torch.pow(self.conditioning_masked - completed_masked, 2))
        
        if self.discriminator:
            self.perceptual = self.discriminator(completed).mean()*1e-2

        self.total_loss = self.content_loss
        if self.discriminator:
            self.total_loss = self.content_loss - self.perceptual

        self.total_loss.backward(retain_graph=True)
        percep_np, content_np, total_np = self.get_numpy_values([-self.perceptual, self.content_loss, self.total_loss])
        self.append_losses([percep_np, content_np, total_np])

        if self.tensorboard is not None:
            if self.perceptual is not None:
                self.tensorboard.add_scalar(scalar_value=percep_np, tag="Perceptual Loss", global_step=self.count[0])
            self.tensorboard.add_scalar(scalar_value=content_np, tag="Content Loss", global_step=self.count[0])
            self.tensorboard.add_scalar(scalar_value=total_np, tag="Total Loss", global_step=self.count[0])

        if self.verbose:
            print("Iteration: ", self.count[0],
                  "Current MSE: %.1f" % float(content_np),
                  " Current Perceptual Loss: %.3f" %float(percep_np),
                  " Current Total Loss: %.3f" %float(total_np),
                  " Target MSE: %.2f"%self.target
                  )
        self.count[0] += 1

        return self.total_loss

    def get_numpy_values(self, variables):
        numpy_values = []
        for v in variables:
            if v is not None:
                numpy_values.append(v.data.cpu().numpy())
            else:
                numpy_values.append(None)
        return numpy_values

    def append_losses(self, losses):
        self.perceptual_losses.append(losses[0])
        self.content_losses.append(losses[1])
        self.total_losses.append(losses[2])
