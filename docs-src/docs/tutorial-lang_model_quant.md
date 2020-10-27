# Post-Training Quantization of a Language Model using Distiller

A detailed, Jupyter Notebook based tutorial on this topic is located at `<distiller_repo_root>/examples/word_language_model/quantize_lstm.ipynb`.  
You can view a "read-only" version of it in the Distiller GitHub repository [here](https://github.com/IntelLabs/distiller/blob/master/examples/word_language_model/quantize_lstm.ipynb).

The tutorial covers the following:

* Converting the model to use Distiller's modular LSTM implementation, which allows flexible quantization of internal LSTM operations.
* Collecting activation statistics prior to quantization
* Creating a `PostTrainLinearQuantizer` and preparing the model for quantization
* "Net-aware quantization" capability of `PostTrainLinearQuantizer`
* Progressively tweaking the quantization settings in order to improve accuracy