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
  * Before running this code, please modify config.py to your own configurations.
  * When training the model with your own data, the dataset should include:
    * image dir: pairs of image patches with transformation parameters (dx, dy, da)
    * pdimage dir: the correspoinding orientation maps
    * menu.txt: in the form of (fname1, fname2, dx, dy, da)
    
* To train the local patch matching model:  
  * Before running this code, please modify config.py to your own configurations.
  * When training the model with your own data, the dataset should include:
    * image dir: image patches centered on key points (minutiae or sampling points). At least 8 images are required for each class.
    * pdimage dir: the correspoinding orientation maps
    * menu.txt: the format of each line is (fname, class_label)
    
Testing
-----
The pretrained patch alignment and patch matching models can be downloaded Baidu Drive (extraction code: ).

* The patch alignment and patch matching algorithms can be tested seperately with the test.py in each dir.
* To obtain the potential corresponding points on a pair of fingerprints, please use the code in Testing dir.


References
-----
Arcface:  https://github.com/ronghuaiyang/arcface-pytorch
Siamese-triplet: https://github.com/adambielski/siamese-triplet
Geometric Matching: https://github.com/hjweide/convnet-for-geometric-matching


Contact
-----

If you have any questions about our work, please contact gus16@mails.tsinghua.edu.cn
