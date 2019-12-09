# Compression Examples

Distiller comes with sample applications and tutorials covering a range of model types:

| Model Type | Sparsity | Post-train quant | Quant-aware training | Auto Compression (AMC) | In Directory |
|------------|:--------:|:----------------:|:--------------------:|:----------------------:|--------------|
| **Image classification** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | [classifier_compression](https://github.com/NervanaSystems/distiller/tree/master/examples/classifier_compression) |
| **Word-level language model** | :white_check_mark: | :white_check_mark: | | |[word_language_model]((https://github.com/NervanaSystems/distiller/tree/master/examples/word_language_model)) |
| **Translation (GNMT)** | | :white_check_mark: | | | [GNMT]((https://github.com/NervanaSystems/distiller/tree/master/examples/GNMT)) |
| **Recommendation System (NCF)** | |  :white_check_mark: | | | [ncf]((https://github.com/NervanaSystems/distiller/tree/master/examples/ncf)) |
| **Object Detection** |  :white_check_mark: | | | | [object_detection_compression]((https://github.com/NervanaSystems/distiller/tree/master/examples/object_detection_compression)) |

The links in the left column in the table point to the code implementing each of the modalities. The rest of the sub-directories in this directory are each dedicated to a specific compression method, and contain YAML schedules and other files that can be used with the sample applications. Most of these files contain details on the results obtained and how to re-produce them.
