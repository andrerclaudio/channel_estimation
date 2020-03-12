import logging

import numpy
import pandas as pd
from scipy.io import loadmat

from models import interpolation, SRCNN_train, SRCNN_predict

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)


def application():
    """" All application has its initialization from here """
    logging.info('Main application is running!')


class ChannelInfo:
    """
    Channel constructor details
    """

    def __init__(self):
        """
        # load dataset
        """
        self.channel_model = "VehA"
        self.SNR = 12
        self.number_of_pilots = 48
        self.perfect = loadmat("Perfect_" + "H_40000" + ".mat")['My_perfect_H']
        self.noisy_input = loadmat("My_noisy_H_12.mat")["My_noisy_H"]


def channel_net_train():
    """
    Deep Learning
    """
    channel_info = ChannelInfo()

    noisy_input = channel_info.noisy_input
    snr = channel_info.SNR
    perfect = channel_info.perfect
    number_of_pilots = channel_info.number_of_pilots
    channel_model = channel_info.channel_model

    interp_noisy = interpolation(noisy_input, snr, number_of_pilots, 'rbf')

    perfect_image = numpy.zeros((len(perfect), 72, 14, 2))
    perfect_image[:, :, :, 0] = numpy.real(perfect)
    perfect_image[:, :, :, 1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(
        2 * len(perfect), 72, 14, 1)

    # ------ training SRCNN ------
    idx_random = numpy.random.rand(len(perfect_image)) < (
            1 / 9)  # uses 32000 from 36000 as training and the rest as validation
    train_data, train_label = interp_noisy[idx_random, :, :, :], perfect_image[idx_random, :, :, :]
    val_data, val_label = interp_noisy[~idx_random, :, :, :], perfect_image[~idx_random, :, :, :]
    SRCNN_train(train_data, train_label, val_data, val_label, channel_model, number_of_pilots, snr)

    # ------ prediction using SRCNN ------
    srcnn_pred_train = SRCNN_predict(train_data, channel_model, number_of_pilots, snr)
    srcnn_pred_validation = SRCNN_predict(train_data, channel_model, number_of_pilots, snr)

    df_train = pd.DataFrame(srcnn_pred_train)
    df_validation = pd.DataFrame(srcnn_pred_validation)

    logger.info(df_train)
    logger.info(df_validation)

    # ------ training DNCNN ------
    # DNCNN_train(input_data, channel_model , Number_of_pilots , SNR):

    return None
