Latent-Fingerprint-Registration-via-Matching-Densely-Sampled-Points
=======
PyTorch implementation of Latent-Fingerprint-Registration-via-Matching-Densely-Sampled-Points


Dependencies
------
* Python 3: cv2, numpy, scipy, matplotlib
* PyTorch >= 1.0
* NVIDIA GPU+CUDA


Training
-----
In the proposed latent fingerprint registration algorithm, the patch alignment and patch matching module are trained seperately. 

* To train  the local patch alignment model:  
  * Before running this code, please modify config.py to your own configurations including:
  * Dataset

Testing
-----

pretrained model

References
-----
Arcface:  https://github.com/ronghuaiyang/arcface-pytorch
Siamese-triplet: https://github.com/adambielski/siamese-triplet
Geometric Matching: https://github.com/hjweide/convnet-for-geometric-matching


Contact
-----
