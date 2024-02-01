# LiteTransNet: An Interpretable Approach for Landslide Displacement Prediction Using Transformer Model with Attention Mechanism

Accurate landslide displacement prediction is crucial for effective early warning systems to mitigate hazards. In this work, we propose LiteTransNet, a Lightweighted Transformer Network tailored for landslide displacement prediction. Built on the revolutionary Transformer model for sequential data, LiteTransNet leverages localized self-attention to selectively focus on relevant timestamps and provides interpretable results through its attention heatmap. Furthermore, LiteTransNet performs end-to-end time series modeling without extensive feature engineering. Our study shows that LiteTransNet innovates landslide displacement prediction by providing accuracy, interpretability, and efficiency, enhancing understanding of deformation mechanisms to facilitate effective early warning systems.

## Getting Started

The instructions below will guide you through the process of setting up and running the LiteTransNet model on your local machine.

### Prerequisites

You will need Python 3.6+ and the following packages:

- PyTorch - version 1.9.0
- Pandas - 1.1.3
- Numpy - 1.21.2
- Matplotlib - 3.3.3
- Scikit-learn - 0.23.2


### Components
- The `data` directory holds the CSV files of the datasets. These files provide the data needed to train the LiteTransNet model.

- The `models` directory stores the saved PyTorch models of the trained Transformer networks. 

- The `tst` directory contains several key components of the transformer architecture used in LiteTransNet.
  
- The script `dataset.py` includes code for handling the dataset class, which preprocesses and provides data to the LiteTransNet model during both training and test stages.

- The script `training.py` manages the training process for the LiteTransNet model. It includes code for handling the training loop and optimization steps, evaluating the test set, and saving checkpoints of the model. 
    
- The script `utils.py` offers general utility functions that assist in preprocessing data.

### Configuration

You can modify the hyperparameters for training in the `param_grid` dictionary within the `training.py` script. Expand `param_grid` to include multiple hyperparameter configurations for grid search. Currently the `param_grid` dictionary includes `d_model`, `q`, `k`, `v`, and `attention_size`.



### Training

The training process involves:

1. Normalizing and reshaping the data into sequences.
2. Defining hyperparameter configurations for grid search.
3. Fitting the LiteTransNet model with specified hyperparameters.
4. Evaluating the predictions and saving the checkpoints of the model. 



## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* https://github.com/maxjcohen/transformer
* https://github.com/jadore801120/attention-is-all-you-need-pytorch