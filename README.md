# pytorch_short_tutorial
This repository describe the short tutorial on Pytorch which will be understandable and meaningful. This could be helpful for beginner who are starting deep learning studies and want to implement those concept using Pytorch. If anyone want to contribute this repository feel free to clone and make a update. thanks

# All about PyTorch

## Short History of PyTorch (Timeline)
* **2016** : Initiated by Adam Paszke as an internship project at Meta AI (then Facebook AI Research), targeting an intuitive Python-based deep learning framework.

* **2017** : Official public release as open source; quickly adopted by the research community due to its Pythonic design and dynamic computation graph.

* **2018** : Merged with Caffe2 to combine research flexibility and production deployment capabilities; release of PyTorch 1.0, adding a stable API and ONNX support.

* **2022** : Transferred governance to the PyTorch Foundation under the Linux Foundation, marking its transition to an industry-standard, community-driven project.

* **2023** : Release of PyTorch 2.0, introducing performance boosters like TorchDynamo for faster execution.

## Why Is PyTorch made it Famous?
* **Pythonic and intuitive** : Designed to look and feel like native Python, making it easily accessible—especially for researchers.

* **Dynamic computation graph** : "Define-by-run" paradigm allows real-time changes and native Python debugging; great for experimentation and rapid prototyping.

* **Excellent documentation and community** : Quickly attracted academics and practitioners, fostering an active code-sharing and open-source culture.

* **Strong industry and research adoption** : Used by tech giants and many AI labs, powering a range of practical applications.

## Benefits Over Other Frameworks
* **Dynamic Over Static Graph** : PyTorch’s dynamic graph allows modifying network structures during runtime—ideal for debugging and developing complex architectures.

* **Easier to learn** : Python-first design minimizes learning curve; seamless integration with Python libraries.

* **Flexibility** : Flexible neural network architectures (such as NLP and reinforcement learning) are easier to build in PyTorch.

* **Superior debugging** : Use of native Python debugging tools, thanks to its dynamic nature.

* **Research to production pipeline** : Merging Caffe2’s strengths made deployment and scalability easier.

## What Makes PyTorch Useful?
* **Fast prototyping and experimentation** : Enables rapid model design and testing, crucial for research environments.

* **Deep Python integration** : Can leverage all of Python’s scientific stack, including NumPy, SciPy, etc..

* **Robust GPU support** : Efficient tensor operations on GPUs, with expanding multi-GPU and distributed support.

* **Transparent and readable code** : Models are easy to understand, debug, and extend—critical for innovation.

* **Wide adoption in research/industry** : Used in computer vision, NLP, recommendation systems, healthcare, robotics, and business scenarios.

## PyTorch vs TensorFlow
| Feature | PyTorch | TensorFlow | 
|:--------|:---------|:------------|
| Computation Graph | Dynamic (define-by-run)| Static (define-then-run)| 
| Syntax/Usability	| More Pythonic; easier for Python users| Steeper learning curve; less Pythonic |
| Debugging	| Native Python debugging, easier| More involved due to static graph |
| Flexibility	| Favored for research, rapid prototyping	| Traditionally for production|
| Production deployment	| Improved since merging with Caffe2| Extensive production tools and APIs| 
| Community & Ecosystem	| Fast-growing, very active	| Larger, mature, more resources| 
| Performance	| Comparable, often faster training	| Comparable, sometimes higher memory usage| 

Both frameworks are powerful. PyTorch is now often preferred for research and experimentation due to its dynamic nature and ease-of-use; TensorFlow remains strong in production settings, but the gap has narrowed sharply in recent years.

## Core Features of pytorch
1. Tensor Computations
2. GPU acceleration
3. Dynamic Computation Graph
4. Automatic Differentiation
5. Distributed Training
6. Interoperability with other libraries

## What Features of Pytorch make it particularly useful for Computer vision and NLP

|Feature|Computer Vision | NLP |
|:------|:---------------|:----|
|Specialized libraries |TorchVision | TorchText, Hugging Face Transformers|
|Pre-trained models| ResNet, VGG, DenseNet, Faster R-CNN | BERT, GPT, custom LSTM/GRU-based model|
|Data handling | DataLoader, image transforms | Tokenization, embeddings, text batching |
| Architecture support | CNNs, GANs, segmentation architectures | RNN, LSTM, GRU, Transformers| 
| Dynamic computation graph | Yes |Yes|
| Deployment | TorchScript, ONNX |TorchScript, ONNX, TorchServe | 
| GPU support | Native | Native |


## PyTorch Core Module

|Module| Description|
|:-----|:-----------|
|```torch```| The core module providing multidimensional arrays(tensor) and mathematical operations on them|
|```torch.autograd```| Automatic differentiation engine that records operations on tensors to compute gradients for optimization|
|```torch.nn``` | Provides a neural networks library, including layers, activations, loss functions and utilities to build deep learning models|
|```torch.optim```| Contains optimization algorithms (optimizer) like SGD, Adam, and DMSprop used for training neural network|
|```torch.utils.data```| Utilities for data handling, including the ```Dataset``` and ```DataLoader``` classes for managing and loading dataset efficiently|
|```torch.jit```| Support Just-In-Time(JIT) compilation and TorchScript for optimizing models and enabling deployment without python dependencies|
|```torch.distributed```| Tools for distributed training across multiple GPUs and machines, facilitating parallel computation|
|```torch.cuda```| Interfaces with NVIDIA CUDA to enable GPU acceleration for tensor computations and model training|
|```torch.backend```| Contains settings and allows control over backend libraries like cuDNN, MKL and others for performance tuning |
|```torch.multiprocessing```| Utilites for parallelism using multiprocessing, similar to python's ```multiprocessing``` module but with support for CUDA tensor|
|```torch.quentization```| Tools for model quantization to reduce model size and improve inference speed, especially on edge devices |
|```torch.onnx```| Supports exporting PyTorch models to the ONNX(Open Neural Network Exchange) format for interoperability with other frameworks and deployment|
