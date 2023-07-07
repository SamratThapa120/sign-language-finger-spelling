import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import math

#Pytorch implementation of https://github.com/hoyso48/tf-utils/blob/main/tf_utils/schedules.py#L4
class OneCycleLR(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer,
                lr=1e-4,
                epochs=10,
                steps_per_epoch=100,
                steps_per_update=1,
                resume_epoch=0,
                decay_epochs=10,
                sustain_epochs=0,
                warmup_epochs=0,
                lr_start=0,
                lr_min=0,
                warmup_type='linear',
                decay_type='cosine',
                last_epoch=-1):
        
        self.lr = float(lr)
        self.epochs = float(epochs)
        self.steps_per_update = float(steps_per_update)
        self.resume_epoch = float(resume_epoch)
        self.steps_per_epoch = float(steps_per_epoch)
        self.decay_epochs = float(decay_epochs)
        self.sustain_epochs = float(sustain_epochs)
        self.warmup_epochs = float(warmup_epochs)
        self.lr_start = float(lr_start)
        self.lr_min = float(lr_min)
        self.decay_type = decay_type
        self.warmup_type = warmup_type
        
        super(OneCycleLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = float(self.last_epoch)
        total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        sustain_steps = self.sustain_epochs * self.steps_per_epoch
        decay_steps = self.decay_epochs * self.steps_per_epoch

        if self.resume_epoch > 0:
            step = step + self.resume_epoch * self.steps_per_epoch

        step = min(decay_steps, step)
        step = (step / self.steps_per_update) * self.steps_per_update

        warmup_cond = step < warmup_steps
        decay_cond = step >= (warmup_steps + sustain_steps)

        lr = self.lr
        if self.warmup_type == 'linear':
            if warmup_cond:
                lr = ((self.lr - self.lr_start) / warmup_steps) * step + self.lr_start
        elif self.warmup_type == 'exponential':
            if warmup_cond:
                factor = self.lr_start ** (1 / warmup_steps)
                lr = (self.lr - self.lr_start) * factor ** (warmup_steps - step) + self.lr_start
        elif self.warmup_type == 'cosine':
            if warmup_cond:
                lr = 0.5 * (self.lr - self.lr_start) * (1 + math.cos(math.pi * (warmup_steps - step) / warmup_steps)) + self.lr_start
        else:
            raise NotImplementedError

        if self.decay_type == 'linear':
            if decay_cond:
                lr = self.lr + (self.lr_min - self.lr) / (decay_steps - warmup_steps - sustain_steps) * (step - warmup_steps - sustain_steps)
        elif self.decay_type == 'exponential':
            if decay_cond:
                factor = self.lr_min ** (1 / (decay_steps - warmup_steps - sustain_steps))
                lr = (self.lr - self.lr_min) * factor ** (step - warmup_steps - sustain_steps) + self.lr_min
        elif self.decay_type == 'cosine':
            if decay_cond:
                lr = 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * (step - warmup_steps - sustain_steps) / (decay_steps - warmup_steps - sustain_steps))) + self.lr_min
        else:
            raise NotImplementedError
        return [lr for base_lr in self.base_lrs]

    def plot(self):
        step = max(1, int(self.epochs*self.steps_per_epoch)//1000) #1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0,int(self.epochs*self.steps_per_epoch),step))
        learning_rates = [self.get_lr()[0] for _ in eps]
        plt.scatter(eps,learning_rates,2)
        plt.show()
