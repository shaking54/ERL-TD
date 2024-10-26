from core.networks import *

actor = DDPGActor("residual", 11, 3, 256, 256, device="cpu")
actor(torch.randn(1, 11))