import sys
import os

import numpy as np
import pymongo


from pymongo import MongoClient
from sklearn import preprocessing

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import conf.settings as config


# data start time and end time
start = config.START
end = config.END


class Data_info_cal():
    """docstring for Data_info_cal.
        it's just prepare some func for std, mean and normalize data
    """

    def __init__(self, data):
        super(Data_info_cal, self).__init__()
        self.start = start
        self.end = end
        self.data = data
        self.close_price = np.array(
            [c['Close'] for c in data.find({
                'Date': {
                    '$lt': end, '$gte': start
                }
            })]
        )

    def get_avg_close_price(self):
        return np.mean(np.array(self.close_price))

    def get_std(self):
        return np.std(self.close_price)

    def get_normalize_data(self):
        return preprocessing.normalize(self.close_price.reshape(1, -1))


if __name__ == '__main__':
    # mongo init
    client = MongoClient(config.MONGO_SERVER, config.MONGO_PORT)
    predict_target = client.predict_target

    # raw data
    NASDAQ_raw = predict_target.Nasdaq
    SOX_raw = predict_target.PHLX_Semiconductor_Index
    SNP500_raw = predict_target.S_and_P500
    DJ_raw = predict_target.Dow_Jones_Industrial_Avg

    d_list = [Data_info_cal(NASDAQ_raw), Data_info_cal(SOX_raw),
              Data_info_cal(SNP500_raw), Data_info_cal(DJ_raw)]
    log_normalize_data = [np.log(d.get_normalize_data()) for d in d_list]

    print([d.get_std() for d in d_list])
    print([d.get_avg_close_price() for d in d_list])
    print(log_normalize_data[0] * log_normalize_data[1]
          * log_normalize_data[2] * log_normalize_data[3])
