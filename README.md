# A Gradient Flow Framework for Analyzing Network Pruning

Codebase for the paper "A Gradient Flow Framework for Analyzing Network Pruning".

## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.4 or higher

To install other dependencies, the following command can be used (uses pip):

```setup
./requirements.sh
```

## Organization
The provided modules serve the following purpose:

* **main.py**: Provides functions for training pruned models.

* **train.py**: Provides functions for training base models.

* **eval.py**: Calculate train accuracy, test accuracy, FLOPs, or compression ratio of a model. Also provides a fine-tuning function.

* **imp_estimator.py**: Importance estimators for different methods.

* **pruner.py**: Pruning engine (includes pruned networks' classes).

* **models.py**: Model classes for VGG-13, MobileNet-V1, ResNet-56 (ResNet-34 and ResNet-18 are also included, but weren't used in the main paper).

* **config.py**: Hyperparameters for training models.

Trained base models will be stored in the directory **pretrained** and pruned models will be saved in **pruned_nets**. Stats collected for train/test numbers are stored in the directory **stats**.

## Example execution 
To prune a model (e.g., resnet-56) using a particular importance measure (e.g., magnitude-based pruning), run the following command

```pruning
python main.py --model=resnet-56 --pruning_type=mag_based --prune_percent=75 --n_rounds=25
```

## Summary of basic options

```--model=<model_name> ```

- *Options*: vgg / mobilenet / resnet-56 / resnet-34 / resnet-18. 

```--seed=<change_random_seed> ```

- *Options*: integer; *Default*: 0.

```--pretrained_path=<use_pretrained_model> ```

- *Options*: string; *Default*: "0". 
- Use this option to prune a partially/fully pretrained model.

```--data_path=<path_to_data> ```

- *Options*: CIFAR10/CIFAR100; *Default*: "CIFAR100". 

```--download=<download_cifar> ```
- *Options*: True/False; *Default*: False.
- If CIFAR-10 or CIFAR-100 are to be downloaded, this option should be True.

```--pruning_type=<how_to_estimate_importance> ```

- *Options*: mag_based / loss_based / biased_loss / grasp / grad_preserve
- The biased_loss measure is biased towards removing small magnitude parameters by default, but the option *use_init* can be used to bias it towards small distance from initialization instead.

```--prune_percent=<how_much_percent_filters_to_prune> ```

- *Options*: integer in range [0, 95]. 
- Target pruning ratio.

```--n_rounds=<number_of_pruning_rounds> ```

- *Options*: integer. 
- Number of rounds Prune+Train is divided over.

```--T=<temperature> ```

- *Options*: float; *Default*: 5.0
- Temperature for training models.

```--grasp_T=<temperature_for_grasp> ```

- *Options*: float; *Default*: 200
- Temperature for calculating Hessian-gradient product.

```--imp_samples=<importance_samples> ```

- *Options*: integer; *Default*: 2560.
- Number of samples to use for importance estimation; default setting corresponds to 20 minibatches.

```--track_stats=<track_train/test_numbers> ```

- *Options*: True/False; *Default*: False.
- Track train/test accuracy for later analysis. 

**Training Settings**: To change number of epochs or the learning rate schedule for training the base models or the pruned models, change the hyperparameters in *config.py*. By default, models are trained using SGD with momentum (0.9).

**Stats**: The stats are stored as a dict divided into train and test, which are both further divided into warmup training, pruning, and final training.

## Evaluation

To evaluate a model (e.g., a pruned VGG-13 model), use:

```eval
python eval.py --model vgg --pruned True --model_path <path_to_model_file> --test_acc True
```

Summary of available options for evaluating models:

```--model=<model_name> ```

- *Options*: vgg/mobilenet/resnet. 

```--pruned=<evaluating_a_pruned_model> ```

- *Options*: True/False; *Default*: False. 
- Set to True for evaluating a pruned model.

```--model_path=<path_to_model> ```

- *Options*: string. 
- Location where model to be analyzed is stored.

```--data_path=<path_to_dataset> ```

- *Options*: string. 
- Location where dataset is stored.
- For CIFAR-10 or CIFAR-100, data_path can be set to 'CIFAR10' or 'CIFAR100', respectively.
- If you are using your own dataset, indicate the path to the main dataset directory which should contain two subdirectories: train and test.

```--train_acc=<evaluate_train_accuracy> ```

- *Options*: True/False; *Default*: False.

```--test_acc=<evaluate_test_accuracy> ```

- *Options*: True/False; *Default*: False.

```--flops=<evaluate_flops_in_model> ```

- *Options*: True/False; *Default*: False.

```--compression=<evaluate_compression_ratio> ```

- *Options*: True/False; *Default*: False.

```--download=<download_standard_dataset> ```

- *Options*: True/False; *Default*: "False". 
- If CIFAR-10 or CIFAR-100 are not already downloaded, they will be downloaded.

```--num_classes=<num_classes> ```

- *Options*: Integer. 
- Number of classes in the dataset used. This has to be set if a dataset other than CIFAR-10 or CIFAR-100 is used.

## Extra functionalities (for experimental purposes)
The codebase contains several functionalities that weren't used in the paper. These allow one to experiment further with our paper's theory. For example, we provide pruned model classes for ResNet-34 and ResNet-18, several other importance measures based on loss-preservation, allow importance tracking over minibatches, options to warmup a model by training for few epochs before pruning, use of manual pruning thresholds, etc. While base settings for these extra functionalities based on our limited tests are already set, we encourage users to fiddle around and engage with us to find better settings or even better importance measures! Here, brief summary of these options is provided:

```--pruning_type=<how_to_estimate_importance> ```

- *Options*: tfo / fisher / nvidia / biased_loss_tracked / l1 / bn
- Some extra importance measures.

```--data_path=<path_to_data> ```

- *Options*: string; *Default*: "CIFAR100". 
- To use a dataset beyond the standard CIFAR datasets, indicate the path to the main dataset directory which should contain two subdirectories: train and test.
- Note: The option num_classes will probably also have to be used with this option.

```--num_classes=<number_of_classes_in_dataset> ```

- *Options*: integer; *Default*: 100. 
- To use a dataset beyond the standard CIFAR datasets, indicate the number of classes in your custom dataset. The model output classifier will be changed accordingly.

```--moment=<momentum_for_training> ```

- *Options*: float; *Default*: 0.9
- Momentum for SGD optimizer.

```--lr_prune=<learning_rate_for_pruning> ```

- *Options*: float; *Default*: 0.1.
- In the prune+train framework, pruning can be chosen to perform at a different learning rate than 0.1.

```--warmup_epochs=<warmup_before_pruning> ```

- *Options*: integer in range [0, 80 - number of pruning rounds]; *Default*: 0.
- To partially train a model before pruning begins, this option can be used.

```--thresholds=<manual_thresholds_for_pruning> ```

- *Options*: array as a string. 
- If you do not want to use the default method for deciding pruning ratios, use this option to define manual thresholds.
- E.g., for pruning the network by 10%, 50%, 80% of original filters in 3 rounds, respectively, use --thresholds='[10, 50, 80]'

```--preserve_moment=<preserve_momentum> ```

- *Options*: True/False; *Default*: False
- In a prune+train framework, a parameter's momentum can be preserved over pruning rounds by using this option.
- This is especially useful when pruning is distributed over a large number of rounds, resulting in small per-round amount of pruning.

```--track_batches=<track_importance> ```

- *Options*: True/False; *Default*: False
- Recent frameworks have shown tracking importance over a few minibatches is helpful. This option enables one to do that for any measure.
- We do not recommend using this with GraSP, as GraSP will take substantial time and memory for its Hessian-gradient product graph.

```--track_batches=<track_batches> ```

- *Options*: integer; *Default*: 20
- If importance is tracked, this option can be used to set the number of minibatches to track importance over. 
- The default option is set to 20 as the untracked importance measures use 20 minibatches to calculate their estimates.

```--alpha=<momentum_for_tracking_importance> ```

- *Options*: Float; *Default*: 0.8
- If importance is tracked, using an exponential average with a momentum-like factor is useful. By default, this hyperparameter is set to 0.8.

```--use_l2=<use_l2> ```

- *Options*: True/False; *Default*: False
- To bias an importance measure towards small magnitude parameters, set this to True.
- Note that the implementation for that measure will have to return a parameter-wise list. An example demonstrating this change is shown using the "biased_loss_tracked" estimator in *imp_estimator.py* file.

```--use_init=<use_init> ```

- *Options*: True/False; *Default*: False
- To bias any of the already implemented importance measures towards small distance from initialization, set this to True.
- Note that the implementation for that measure will have to return a parameter-wise list. An example demonstrating this change is shown using the "biased_loss_tracked" estimator in *imp_estimator.py* file.

```--bypass=<bypass> ```

- *Options*: True/False; *Default*: False
- To bypass any of the default settings and choose your own setting, use this option.