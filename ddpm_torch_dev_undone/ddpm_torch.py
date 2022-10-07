from dev.ddpm_functions import load_data, UniformSampler
from dev.ddpm_functions import create_argparser, args_to_dict
from dev.ddpm_functions import create_model_and_diffusion, model_and_diffusion_defaults
from dev.ddpm_functions import zero_grad

args = create_argparser().parse_args()
data = load_data(data_dir="./", batch_size=1, image_size=64)
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
schedule_sampler = UniformSampler(diffusion)

microbatch = 4
batch_shape0 = 16

for epoch in range(100):
    batch, cond = next(data)
    zero_grad(list(model.parameters()))
    for i in range(0, batch_shape0, microbatch):
        micro = batch[i : i + microbatch]
        micro_cond = {k: v[i : i+microbatch] for k, v in cond.items()}
        last_batch = (i + microbatch) >= batch_shape0
        t, weights = schedule_sampler.sample(microbatch)
        diffusion.training_losses(model, micro, t, model_kwargs=micro_cond)

def forward_backward(self, batch, cond):
    
    for i in range(0, batch.shape[0], self.microbatch):

        if last_batch or not self.use_ddp:
            losses = compute_losses()
        else:
            with self.ddp_model.no_sync():
                losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()