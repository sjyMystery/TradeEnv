
"""
    DATA FEED FORMAT

    ( start_time,period,ask_open,ask_close,ask_low,ask_high,bid_open,bid_close,bid_low,bid_high )

    - start_time : time in UNIX seconds
    - period : second
"""


from tradenv.Feed.BasicFeed import BasicFeed
from tradenv.Feed.CSVFeed import CSVFeed