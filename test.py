import torch
import os
import config_dqn as config
resume_path = os.path.join("log", config.task, config.resume_model)
ckpt_path = os.path.join(resume_path, "Track_train.pth")
checkpoint = torch.load(ckpt_path)
print(checkpoint['high_level_policy'])
