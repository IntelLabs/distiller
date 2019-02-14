from .automl_args import add_automl_args

def do_adc(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn):
    from .ADC import do_adc_internal
    do_adc_internal(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn)
