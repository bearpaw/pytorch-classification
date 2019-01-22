import torch.optim.lr_scheduler as lr_scheduler
import math 

class GradientRatioScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, smooth=0, decay_factor=1):
        self.lr_factors = [1 for _ in optimizer.param_groups]
        self.smooth = smooth
        self.decay_factor = decay_factor
        self.cached_lrs = None
        super(GradientRatioScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.cached_lrs is None:
            self.cached_lrs = self.get_rat_mul_lr()
        return self.cached_lrs

    def blend_lr(self, lr0, factor, alpha):
        return alpha*lr0 + (1-alpha)*(lr0*factor)
    def get_rat_blended_lr(self):
        epoch = max(self.last_epoch, 0)
        alpha = min(epoch/10.0, 1.0)
        return [self.blend_lr(base_lr * self.decay_factor, self.lr_factors[i], alpha) for i,base_lr in enumerate(self.base_lrs)]

    def get_rat_mul_lr(self):
        return [(base_lr * self.decay_factor) * self.lr_factors[i] for i,base_lr in enumerate(self.base_lrs)]

    def get_rat_exp_disc_lr(self):
        x = max(self.last_epoch, 1e-10) / 8
        return [(base_lr * self.decay_factor) * \
            (self.lr_factors[i] / math.exp(x) + math.exp(-1.0/x)) \
            for i,base_lr in enumerate(self.base_lrs)]

    def get_decay_factor(self):
        return self.decay_factor
    def set_decay_factor(self, decay_factor):
        self.decay_factor = decay_factor
        self.cached_lrs = None

    def on_after_batch(self):
        this_g_sum, this_g_count = 0, 0
        last_g = None

        i, m_i = 0, None
        i_start = None
        pgs = list(self.optimizer.param_groups)
        while i < len(pgs):
            pg = pgs[i]
            if m_i is None:
                m_i = pg['m_i']
            #TODO remove dep on m_i?
            if pg['m_i'] != m_i: # new module started
                # save last moddule's avg
                if last_g is None and this_g_count > 0:
                    last_g = this_g_sum / this_g_count
                this_g_sum, this_g_count = 0, 0
                m_i = pg['m_i']
                i_start = i

            # accumulate for current module
            #TODO - worry about requires_grad
            for p in pg['params']:
                if p.requires_grad and p.grad is not None and p.grad.numel() > 0:
                    this_g_sum += p.grad.abs().sum().item()
                    this_g_count += p.grad.numel()
            
            # if this is the last one in module, update LR
            if last_g is not None and i_start is not None and this_g_count >0 and \
                (i == len(pgs)-1 or pgs[i+1]['m_i'] != m_i):

                rat = last_g / (this_g_sum / this_g_count)

                #rat_sig = math.tanh(rat)
                if math.isnan(rat):
                    rat = 1
                rat = max(min(rat, 20.0), 0.1)

                #if rat < 0.1 or rat > 10 or math.isnan(rat):
                #    print(rat, m_i, i)
                for j in range(i, i_start-1, -1):
                    self.lr_factors[j] = \
                        self.smooth*self.lr_factors[j] + (1-self.smooth)*rat
            i += 1
        self.cached_lrs = None

    @staticmethod
    def get_named_members(model, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = model.named_modules(prefix=prefix) if recurse else [(prefix, model)]
        for i, (module_prefix, module) in enumerate(modules):
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    print("WARNING: reused module parameter")
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v, i

    @staticmethod
    def named_parameters(model, prefix='', recurse=True):
        gen = GradientRatioScheduler.get_named_members(model,
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for name, p, i in gen:
            yield name, p, i

    @staticmethod
    def get_params_base_lr(model, lr):
        param_lr=[]
        for name, p, i in GradientRatioScheduler.named_parameters(model):
            param_lr.append({'params': p, 'm_i':i, 'lr': lr})
        param_lr = list(reversed(param_lr)) # we start from last layer
        return param_lr