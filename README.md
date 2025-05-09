# A Primal-Dual Perspective for Distributed TD-Learning  
[![arXiv](https://img.shields.io/badge/arXiv-2310.00638-b31b1b.svg)](https://arxiv.org/abs/2310.00638)

This repository contains the official implementation of the paper  
**"A Primal-Dual Perspective for Distributed TD-Learning"**  
To appear in *IJCAI 2025 (Main Track)*.  
[[arXiv:2310.00638](https://arxiv.org/abs/2310.00638)]


### Running the experiments
- For single fixed paramers, run ```python single_exp.py```.
- For multiple runs for single fixed parameters, run ```python multiple_exp.py```.



### Configuration files

- The agent configs are in agnets/configs directory. (learning rate or etc)
- Other parameters (e.g., algorithm and graph type) can be changed in ```/utils/parser_utils.py```