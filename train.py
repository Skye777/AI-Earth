"""
@author: Skye Cui
@file: train.py
@time: 2021/2/20 16:35
@description: 
"""
import os
os.environ["LOGURU_INFO_COLOR"] = "<green>"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import time
import re
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from loguru import logger
from progress.spinner import MoonSpinner
from input import *
from model import UTransformer
from loss import Loss
from metric import score, nino_seq

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def main():
    train_dataset, test_dataset = train_input_fn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)
    model = UTransformer(hp)
    model_loss = Loss(model)

    best_score = float('-inf')
    not_improved_count = 0

    checkpoint_file = hp.ckpt
    if checkpoint_file == '':
        checkpoint_file = 'ckp_0'
    else:
        model.load_weights(f'{hp.single_gpu_model_dir}/{checkpoint_file}')

    logger.add(f"{hp.logdir}/cmip_train.log", enqueue=True)

    for epoch in range(hp.num_epochs):
        for step, (x_batch_train, ys_batch_train) in enumerate(train_dataset):
            start = time.clock()
            with tf.GradientTape() as tape:
                y_predict = model([x_batch_train, ys_batch_train], training=True)
                loss_ssim, loss_l2, loss_l1, loss = model_loss([y_predict, ys_batch_train[1]])
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            elapsed = (time.clock() - start)
            template = ("step {} loss is {:1.5f}, "
                        "loss ssim is {:1.5f}, "
                        "loss l2 is {:1.5f}, "
                        "loss l1 is {:1.5f}."
                        "({:1.2f}s/step)")
            logger.info(template.format(step, loss.numpy(), loss_ssim.numpy(), loss_l2.numpy(), loss_l1.numpy(), elapsed))

        if epoch % hp.num_epoch_record == 0:
            loss_test = 0
            loss_ssim_test = 0
            loss_l2_test = 0
            loss_l1_test = 0
            count = 0
            y_true, y_pred = [], []
            spinner = MoonSpinner('Testing ')
            for step, (x_batch_test, ys_batch_test) in enumerate(test_dataset):
                y_predict = model([x_batch_test, ys_batch_test], training=False)
                loss_ssim, loss_l2, loss_l1, loss = model_loss([y_predict, ys_batch_test[1]])
                loss_ssim_test += loss_ssim.numpy()
                loss_l2_test += loss_l2.numpy()
                loss_l1_test += loss_l1.numpy()
                loss_test += loss.numpy()
                count += 1

                y_true.append(np.array(nino_seq(ys_batch_test[1][:, :, :, :, 0])))
                y_pred.append(np.array(nino_seq(y_predict[:, :, :, :, 0])))

                spinner.next()

            y_true = tf.concat(y_true, axis=0)
            y_pred = tf.concat(y_pred, axis=0)
            sco = score(y_true, y_pred)
            if sco > best_score:
                best_score = sco
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            spinner.finish()
            logger.info("TEST COMPLETE!")
            template = ("TEST DATASET STATISTICS: "
                        "loss is {:1.5f}, "
                        "loss ssim is {:1.5f}, "
                        "loss l2 is {:1.5f}, "
                        "loss l1 is {:1.5f},"
                        "acc skill score is {:1.5f}.")
            logger.info(template.format(loss_test/count, loss_ssim_test/count, loss_l2_test/count, loss_l1_test/count, sco))

            total_epoch = int(re.findall("\d+", checkpoint_file)[0])
            checkpoint_file = checkpoint_file.replace(f'_{total_epoch}', f'_{total_epoch + 1}')
            # if not_improved_count == hp.early_stop_patience:
            #     print("Validation performance didn\'t improve for {} epochs. "  "Training stops.".format(
            #         hp.early_stop_patience))
            #     break
            # if best_state:
            model.save_weights(f'{hp.single_gpu_model_dir}/{checkpoint_file}', save_format='tf')
            # model.save("my_model")
            logger.info("Saved checkpoint_file {}".format(checkpoint_file))


if __name__ == '__main__':
    main()

