from decimal import Decimal

class Pair_Trading_Config:
    # Target symbol is where the LIMIT orders are placed based on calculated spreads.
    REFERENCE_SYMBOL = "ETHUSDT"
    TARGET_SYMBOL = "SOLUSDT"

    # Reference symbol is where the MARKET orders are intiated AFTER target symbol's limit orders are filled.
    
   #TARGET_SYMBOL = "ETH_USDT"

    # Reference symbol is where the MARKET orders are intiated AFTER target symbol's limit orders are filled.
    #REFERENCE_SYMBOL = "BTC_USDT"

    OPEN_THRESHOLD = 1.5

    STOP_LOSS_THRESHOLD = 20
    # Window size for calculating spread mean.
    MA_WINDOW_SIZE = 100
    ˋ
    RETRY_TIME = 1
    
    PRECISION_AMOUNT_REF = Decimal('0.000')
    
    PRECISION_PRICE_REF = Decimal('0.00')
    
    
    PRECISION_AMOUNT_TARGET = Decimal('0.000')
    
    PRECISION_PRICE_TARGET = Decimal('0')
    
    SLIPPAGE = 0.001
    TEST_SECOND = 300
    CAPITAL = 2000
    POS_RATIO = 1600
    HOLD_DAY = 7
    