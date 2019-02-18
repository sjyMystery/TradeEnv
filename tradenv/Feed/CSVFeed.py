from tradenv.Feed.BasicFeed import BasicFeed
import numpy as np
import pandas as pd
import time


class CSVFeed(BasicFeed):

    def append_csv(self, file_name):
        df = pd.read_csv(file_name)
        df["DateTime"] = df["DateTime"].apply(pd.to_datetime).apply(lambda x: time.mktime(x.timetuple()) / 60)
        df = df[["AskOpen", "AskHigh", "AskLow", "AskClose", "BidOpen", "BidHigh", "BidLow", "BidClose","DateTime"]]
        data = np.array(df)
        self.append(data)

    def append_csv_list(self, file_list):
        for name in file_list:
            self.append(name)
