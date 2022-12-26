import sys
import os
import os,sys
#設定當前工作目錄，放再import其他路徑模組之前
os.chdir(sys.path[0])
sys.path.append('./simulator')
sys.path.append('./module')
from TradingSimulator import TradingSimulator

trading_simulator = TradingSimulator(
    #  hive_host='hive_host',
    #  hive_port='hive_port',
    #  hive_database='hive_database',
    #  hive_username='hive_username',
    #  hive_password='hive_password'
    hive_host='10.107.22.111',
    hive_port='10000',
    hive_database='btse',
    hive_username='datateam',
    hive_password='iDogXdijd/Zm/P'
)

def main():
    from module.spreader import Spreader
    from config import Pair_Trading_Config
    config = Pair_Trading_Config()
    spreader = Spreader(None, config, mode='staging')
    # the data structure of result is the same with handler argument
    result = trading_simulator.simulate(
        exchange_symbols={
           'BINANCE': [config.REFERENCE_SYMBOL, config.TARGET_SYMBOL]
        },
        start_time='2022-02-01T00:00:00.000Z',
        end_time='2022-02-05T00:00:00.000Z',
        handler=spreader.simulate_handler,
        #handler = handler,
        init_metadata={ 'count': 0 }
    )


if __name__ == '__main__':
    main()
