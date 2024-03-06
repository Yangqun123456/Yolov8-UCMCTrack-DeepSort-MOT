# icon7.addFile(u":/icons/images/icons/cil-media-pause.png", QSize(), QIcon.Normal, QIcon.On)

# from .resources_rc import *

import torch

if torch.cuda.is_available():
    print("PyTorch can use GPUs!")
else:
    print("PyTorch cannot use GPUs.")