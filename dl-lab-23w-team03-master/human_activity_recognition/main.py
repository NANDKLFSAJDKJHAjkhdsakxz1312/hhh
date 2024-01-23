import gin
import logging
from absl import app, flags
import  tensorflow as tf
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models import create_model

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = datasets.load()
    print(ds_train)
    for window_sequence, labels in ds_train.take(1):
        tf.print("Features:", window_sequence)
        tf.print("Labels:", labels)

    # model
    model = create_model(input_shape=(250,6), num_classes=12)

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    '''else:
        evaluate(model,
                 run_paths,
                 ds_test)'''

if __name__ == "__main__":
    app.run(main)
