# Compression Examples

Distiller comes with sample applications and tutorials covering a range of model types:

| Model Type | Sparsity | Post-train quant | Quant-aware training | Auto Compression (AMC) |
|------------|:--------:|:----------------:|:--------------------:|:----------------------:|
| [Image classification](https://github.com/NervanaSystems/distiller/tree/master/examples/classifier_compression) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Word-level language model](https://github.com/NervanaSystems/distiller/tree/master/examples/word_language_model)| :white_check_mark: | :white_check_mark: | | |
| [Translation (GNMT)](https://github.com/NervanaSystems/distiller/tree/master/examples/GNMT) | | :white_check_mark: | | |
| [Recommendation System (NCF)](https://github.com/NervanaSystems/distiller/tree/master/examples/ncf) | |  :white_check_mark: | | |
| [Object Detection](https://github.com/NervanaSystems/distiller/tree/master/examples/object_detection_compression) |  :white_check_mark: | | | |

The links in the left column in the table point to the code implementing each of the modalities. The rest of the sub-directories in this directory are each dedicated to a specific compression method, and contain YAML schedules and other files that can be used with the sample applications. Most of these files contain details on the results obtained and how to re-produce them.
