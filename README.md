# neuralnets-semantics

This repo is implementing Word representation as feature vectors in :

* Python with numpy - DONE
* Python with Theano - DONE (but need some clean-up)
* Python with TensorFlow - DONE (but need some clean-up)
* Python with Keras - to do
* Python with PyTorch - to do


The model is based on a neural network that is trained on predicting the last word in a 4-gram.
The plain Python version is based on an original Octave code from *G Hinton*.



## AWS EC2 configuration

These scripts are run on AWS EC2 GPU "g2.2x" instance based on AMI (Ireland) :
    
cs231n_caffe_torch7_keras_lasagne_v2 **ami-e8a1fe9b**
    
At EC2 configuration time, to setup Jupyter web I follow this tutorial :
    
http://efavdb.com/deep-learning-with-jupyter-on-aws/
    
To re-use the same folder across multiple EC2 launches I use AWS EFS :
```
($ sudo apt-get update ?)
$ sudo apt-get -y install nfs-common
($ sudo reboot ?)
$ cd caffe
$ mkdir neuralnets
$ cd ..
$ sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone).YOUR_EFS_HERE.efs.YOUR_ZONE_HERE.amazonaws.com:/ caffe/neuralnets
($ clone Git repo in neuralnets directory ?)
```
Note : the security group of the EFS folder and EC2 instace needs to be configured correctly :

http://docs.aws.amazon.com/efs/latest/ug/accessing-fs-create-security-groups.html


The EC2 AMI comes with Theano but TensorFlow needs to be installed :
```
($ easy_install --upgrade pip ?)
$ pip install tensorflow
```


WARNING : With this setup Theano makes use of the GPU but TensorFlow only runs on the CPU


To run Theano script with GPU :
```
$ cd caffe/neuralnets/nb_theano
$ THEANO_FLAGS='floatX=float32,device=gpu' python dA.py
```


To run Keras you may have to un-install/re-install.
You may have to edit the Keras json file to switch between Theano (th) and TensorFlow (tf) backends.


To unmount the EFS folder before closing down the EC2 instance :
```
$ sudo umount caffe/neuralnets
```