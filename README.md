# Multimodal segmentation with disentangled representations


Implementations of the **MMSDNet** and **DAFNet** models that perform multimodal  image segmentation using a disentangled representation of anatomy and modality factors. For further details please see our paper [Multimodal Cardiac Segmentation Using Disentangled Representation Learning] presented in STACOM 2019, and the pre-print [Disentangle Align and Fuse for Multimodal and Zero-shot Image Segmentation].

The code is written in Python 3.6 with [Keras] 2.1.6 and [tensorflow] 1.4.0 and 
experiments were run in Titan-X and Titan-V GPUs. The `requirements.txt` file contains all Python library versions.

This project is structured in different packages as follows:

* **callbacks**: contains Keras callbacks for printing images and losses during training
* **configuration**: contains configuration files for running an experiment
* **layers**: contains custom Keras layers, e.g. the STN layer
* **loaders**: contains definitions of data loaders
* **model_components**: contains implementations of individual components, e.g. the encoders and decoders
* **model_executors**: contains code for loading data and training models
* **models**: contains Keras implementations of MMSDNet, DAFNet
* **utils**: package with utility functions used in the project

To define a new data loader, extend class `base_loader.Loader`, and register the loader in `loader_factory.py`. The datapaths are specified in `base_loader.py`.

To run an experiment, execute `experiment.py`, passing the configuration filename and the split number as runtime parameters:
```
python experiment.py --config myconfiguration --split 0
```
Optional parameters include the proportion of labels in modality 2 data for unsupervised learning, e.g. 
```
python experiment.py --config myconfiguration --split 0 --l_mix 0.5
```

Sample config files for MMSDNet and DAFNet are placed in the **configuration** package for [CHAOS] data. 

## Citation

If you use this code for your research, please cite our papers:

```
@InProceedings{chartsias2020multimodal,
author="Chartsias, Agisilaos and Papanastasiou, Giorgos and Wang, Chengjia and Stirrat, Colin and Semple, Scott and Newby, David and Dharmakumar, Rohan and Tsaftaris, Sotirios A.",
title="Multimodal Cardiac Segmentation Using Disentangled Representation Learning",
booktitle="Statistical Atlases and Computational Models of the Heart. Multi-Sequence CMR Segmentation, CRT-EPiggy and LV Full Quantification Challenges",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="128--137",
isbn="978-3-030-39074-7"
}
```

```
@article{chartsias2020disentangle,
  title={Disentangle, align and fuse for multimodal and zero-shot image segmentation},
  author={Chartsias, Agisilaos and Papanastasiou, Giorgos and Wang, Chengjia and Semple, Scott and Newby, David and Dharmakumar, Rohan and Tsaftaris, Sotirios A},
  journal={arXiv preprint arXiv:1911.04417},
  year={2019}
}

```

[Multimodal Cardiac Segmentation Using Disentangled Representation Learning]: https://link.springer.com/chapter/10.1007/978-3-030-39074-7_14
[Disentangle Align and Fuse for Multimodal and Zero-shot Image Segmentation]: https://arxiv.org/abs/1911.04417
[Keras]: https://keras.io/
[tensorflow]: https://www.tensorflow.org/
[CHAOS]: http://doi.org/10.5281/zenodo.3362844