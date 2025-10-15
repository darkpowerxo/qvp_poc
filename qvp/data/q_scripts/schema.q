/ schema.q - Database Schema Definitions for Market Data
/ Optimized table structures for tick data and analytics

/ ============================================================================
/ TICK DATA SCHEMA
/ ============================================================================

/ Main tick table - stores every trade
ticks:([] 
  time:`timestamp$();      / Trade timestamp (nanosecond precision)
  sym:`symbol$();          / Symbol/ticker
  price:`float$();         / Trade price
  size:`long$();           / Trade size (shares/contracts)
  exchange:`symbol$();     / Exchange code (NASDAQ, NYSE, etc.)
  conditions:`symbol$()    / Trade conditions/flags
);

/ Create indices for fast queries
`.Q.ind[`ticks;`sym`time]

/ Quote table - stores bid/ask data
quotes:([]
  time:`timestamp$();
  sym:`symbol$();
  bid:`float$();
  bidsize:`long$();
  ask:`float$();
  asksize:`long$();
  exchange:`symbol$()
);

/ ============================================================================
/ AGGREGATED BAR DATA
/ ============================================================================

/ OHLCV bars - time-aggregated data
ohlcv:([]
  time:`timestamp$();      / Bar timestamp
  sym:`symbol$();
  open:`float$();
  high:`float$();
  low:`float$();
  close:`float$();
  volume:`long$();
  vwap:`float$();          / Volume-weighted average price
  trades:`long$()          / Number of trades in bar
);

/ Daily bars with extended metrics
daily:([]
  date:`date$();
  sym:`symbol$();
  open:`float$();
  high:`float$();
  low:`float$();
  close:`float$();
  volume:`long$();
  adjclose:`float$();      / Adjusted close
  dividends:`float$();
  splits:`float$()
);

/ ============================================================================
/ VOLATILITY TABLES
/ ============================================================================

/ Realized volatility calculations
realized_vol:([]
  date:`date$();
  sym:`symbol$();
  rv_5min:`float$();       / 5-minute realized vol
  rv_1hour:`float$();      / 1-hour realized vol  
  rv_daily:`float$();      / Daily realized vol
  parkinson:`float$();     / Parkinson estimator
  garman_klass:`float$();  / Garman-Klass estimator
  rogers_satchell:`float$();
  yang_zhang:`float$()     / Yang-Zhang estimator
);

/ Implied volatility (for options data)
implied_vol:([]
  date:`date$();
  sym:`symbol$();
  expiry:`date$();
  strike:`float$();
  call_iv:`float$();
  put_iv:`float$();
  option_type:`symbol$()
);

/ VIX and volatility indices
vol_indices:([]
  time:`timestamp$();
  vix:`float$();           / CBOE VIX
  vxn:`float$();           / CBOE NASDAQ VIX
  vvix:`float$();          / VIX of VIX
  skew:`float$()           / CBOE SKEW index
);

/ ============================================================================
/ STRATEGY TABLES
/ ============================================================================

/ Portfolio positions
positions:([]
  time:`timestamp$();
  sym:`symbol$();
  quantity:`long$();
  entry_price:`float$();
  current_price:`float$();
  pnl:`float$();
  pnl_pct:`float$()
);

/ Trade signals
signals:([]
  time:`timestamp$();
  sym:`symbol$();
  signal:`symbol$();       / `buy, `sell, `hold
  strategy:`symbol$();     / Strategy name
  strength:`float$();      / Signal strength
  metadata:`symbol$()      / Additional parameters
);

/ Execution records
executions:([]
  time:`timestamp$();
  order_id:`symbol$();
  sym:`symbol$();
  side:`symbol$();         / `buy or `sell
  quantity:`long$();
  price:`float$();
  commission:`float$();
  slippage:`float$();
  status:`symbol$()        / `filled, `partial, `rejected
);

/ ============================================================================
/ RISK METRICS
/ ============================================================================

/ Portfolio risk metrics
risk_metrics:([]
  date:`date$();
  portfolio_value:`float$();
  var_95:`float$();        / Value at Risk (95%)
  cvar_95:`float$();       / Conditional VaR
  sharpe:`float$();
  sortino:`float$();
  max_drawdown:`float$();
  beta:`float$()
);

/ Stress test results
stress_tests:([]
  date:`date$();
  scenario:`symbol$();
  portfolio_impact:`float$();
  var_impact:`float$();
  largest_loss:`symbol$()  / Symbol with largest loss
);

/ ============================================================================
/ HELPER FUNCTIONS
/ ============================================================================

/ Initialize all tables
initTables:{[]
  / Reset all tables to empty
  `ticks set ([] time:`timestamp$(); sym:`symbol$(); price:`float$(); size:`long$(); exchange:`symbol$(); conditions:`symbol$());
  `quotes set ([] time:`timestamp$(); sym:`symbol$(); bid:`float$(); bidsize:`long$(); ask:`float$(); asksize:`long$(); exchange:`symbol$());
  `ohlcv set ([] time:`timestamp$(); sym:`symbol$(); open:`float$(); high:`float$(); low:`float$(); close:`float$(); volume:`long$(); vwap:`float$(); trades:`long$());
  `daily set ([] date:`date$(); sym:`symbol$(); open:`float$(); high:`float$(); low:`float$(); close:`float$(); volume:`long$(); adjclose:`float$(); dividends:`float$(); splits:`float$());
  `realized_vol set ([] date:`date$(); sym:`symbol$(); rv_5min:`float$(); rv_1hour:`float$(); rv_daily:`float$(); parkinson:`float$(); garman_klass:`float$(); rogers_satchell:`float$(); yang_zhang:`float$());
  `implied_vol set ([] date:`date$(); sym:`symbol$(); expiry:`date$(); strike:`float$(); call_iv:`float$(); put_iv:`float$(); option_type:`symbol$());
  `vol_indices set ([] time:`timestamp$(); vix:`float$(); vxn:`float$(); vvix:`float$(); skew:`float$());
  `positions set ([] time:`timestamp$(); sym:`symbol$(); quantity:`long$(); entry_price:`float$(); current_price:`float$(); pnl:`float$(); pnl_pct:`float$());
  `signals set ([] time:`timestamp$(); sym:`symbol$(); signal:`symbol$(); strategy:`symbol$(); strength:`float$(); metadata:`symbol$());
  `executions set ([] time:`timestamp$(); order_id:`symbol$(); sym:`symbol$(); side:`symbol$(); quantity:`long$(); price:`float$(); commission:`float$(); slippage:`float$(); status:`symbol$());
  `risk_metrics set ([] date:`date$(); portfolio_value:`float$(); var_95:`float$(); cvar_95:`float$(); sharpe:`float$(); sortino:`float$(); max_drawdown:`float$(); beta:`float$());
  `stress_tests set ([] date:`date$(); scenario:`symbol$(); portfolio_impact:`float$(); var_impact:`float$(); largest_loss:`symbol$());
  };

/ Get table schema
getSchema:{[tableName]
  meta tableName
  };

/ Get table count
getCount:{[tableName]
  count value tableName
  };

/ Get table memory usage (approximate)
getMemory:{[tableName]
  -22!value tableName  / Memory usage in bytes
  };

/ Optimize table (apply attributes)
optimizeTable:{[tableName]
  / Apply sorted attribute to time column
  @[tableName;`time;`s#];
  / Apply grouped attribute to sym column  
  @[tableName;`sym;`g#];
  };

/ ============================================================================
/ PARTITIONED DATABASE SETUP
/ ============================================================================

/ Create partitioned database on disk
createPartitionedDB:{[dbPath]
  / dbPath: directory path for database
  
  / Create directory if not exists
  system "mkdir -p ",dbPath;
  
  / Set database
  `:dbPath set `.;
  
  / Create partitioned tables (by date)
  / This allows efficient queries on large datasets
  `:dbPath/ticks/ set .Q.en[`:dbPath] ticks;
  `:dbPath/ohlcv/ set .Q.en[`:dbPath] ohlcv;
  `:dbPath/daily/ set .Q.en[`:dbPath] daily;
  
  / Load database
  system "l ",dbPath;
  };

/ ============================================================================
/ DATA VALIDATION
/ ============================================================================

/ Validate tick data
validateTicks:{[tickData]
  / Check for nulls
  nulls:sum each null flip tickData;
  
  / Check for negative prices
  negPrices:sum tickData[`price]<0;
  
  / Check for zero sizes
  zeroSizes:sum tickData[`size]=0;
  
  / Check time ordering
  timeOrder:all (<=) tickData[`time];
  
  ([] 
    check:`nulls`negPrices`zeroSizes`timeOrder;
    count:(sum nulls;negPrices;zeroSizes;not timeOrder);
    status:(`OK`ERROR`ERROR`ERROR)
  )
  };

/ Clean tick data
cleanTicks:{[tickData]
  / Remove nulls
  tickData:delete from tickData where null price;
  tickData:delete from tickData where null size;
  
  / Remove invalid prices
  tickData:delete from tickData where price<=0;
  
  / Remove zero sizes
  tickData:delete from tickData where size=0;
  
  / Sort by time
  tickData:`time xasc tickData;
  
  tickData
  };

/ ============================================================================
/ EXAMPLE USAGE
/ ============================================================================

/ Initialize and show table info
showTableInfo:{[]
  initTables[];
  
  tables:tables[];
  
  ([]
    table:tables;
    columns:count each meta each tables;
    rows:getCount each tables
  )
  };
