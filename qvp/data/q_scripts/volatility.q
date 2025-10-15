/ volatility.q - Advanced Volatility Calculations in q
/ High-performance implementations of volatility estimators

/ ============================================================================
/ REALIZED VOLATILITY
/ ============================================================================

/ Calculate realized variance from high-frequency returns
/ Uses sum of squared intraday returns
realizedVar:{[prices;freq]
  / prices: vector of prices
  / freq: sampling frequency (e.g., `minute, `5minute, `hour)
  / returns: 252 * sum(returns^2) - annualized
  
  logrets:1_deltas log prices;
  rv:sum logrets*logrets;
  252*rv
  };

/ Calculate realized volatility with time bars
realizedVolBars:{[tickData;interval]
  / tickData: table with time, sym, price columns
  / interval: time interval (e.g., 0D00:01:00 for 1 minute)
  
  / Create time-barred prices
  bars:select last price by interval xbar time from tickData;
  
  / Compute log returns
  bars:update logret:log price % prev price from bars;
  
  / Realized variance
  rv:sum (bars`logret)*bars`logret;
  sqrt 252*rv
  };

/ ============================================================================
/ PARKINSON VOLATILITY ESTIMATOR
/ Uses high-low range
/ ============================================================================

parkinsonVol:{[high;low]
  / Parkinson (1980) estimator
  / sigma^2 = (1/(4*n*ln2)) * sum(ln(H/L)^2)
  
  n:count high;
  hl_ratio:log high%low;
  sigma2:(sum hl_ratio*hl_ratio)%(4*n*log 2);
  sqrt sigma2
  };

/ ============================================================================
/ GARMAN-KLASS VOLATILITY ESTIMATOR
/ Uses OHLC data
/ ============================================================================

garmanKlassVol:{[open;high;low;close]
  / Garman-Klass (1980) estimator
  / More efficient than close-to-close
  
  n:count open;
  hl:log high%low;
  co:log close%open;
  
  / GK formula: 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
  term1:0.5*hl*hl;
  term2:(2*log 2;1)*co*co;
  
  sigma2:(sum term1-term2)%n;
  sqrt sigma2
  };

/ ============================================================================
/ ROGERS-SATCHELL VOLATILITY ESTIMATOR
/ Drift-independent estimator
/ ============================================================================

rogersSatchellVol:{[open;high;low;close]
  / Rogers-Satchell (1991) estimator
  / Handles drift without bias
  
  n:count open;
  hc:log high%close;
  ho:log high%open;
  lc:log low%close;
  lo:log low%open;
  
  / RS formula: ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
  rs_terms:(hc*ho)+(lc*lo);
  sigma2:(sum rs_terms)%n;
  sqrt sigma2
  };

/ ============================================================================
/ YANG-ZHANG VOLATILITY ESTIMATOR
/ Most efficient OHLC estimator (14x better than close-to-close)
/ ============================================================================

yangZhangVol:{[open;high;low;close]
  / Yang-Zhang (2000) estimator
  / Combines overnight, open-to-close, and Rogers-Satchell
  
  n:count open;
  
  / Overnight volatility
  co:log open%prev close;
  overnight_var:(sum co*co)%(n-1);
  
  / Open-to-close volatility  
  oc:log close%open;
  oc_mean:avg oc;
  oc_centered:oc-oc_mean;
  openclose_var:(sum oc_centered*oc_centered)%(n-1);
  
  / Rogers-Satchell component
  hc:log high%close;
  ho:log high%open;
  lc:log low%close;
  lo:log low%open;
  rs_terms:(hc*ho)+(lc*lo);
  rs_var:(sum rs_terms)%n;
  
  / Combine components (k=0.34 is optimal)
  k:0.34;
  yz_var:overnight_var + k*openclose_var + (1-k)*rs_var;
  
  sqrt yz_var
  };

/ ============================================================================
/ INTRADAY PATTERNS
/ ============================================================================

/ Extract intraday volatility pattern
intradayVolPattern:{[tickData;interval]
  / tickData: table with time, price
  / interval: time bucket (e.g., 0D00:05:00 for 5-min)
  
  / Get time of day
  tickData:update tod:time.time from tickData;
  
  / Compute returns by time bucket
  bars:select last price by interval xbar tod from tickData;
  bars:update ret:log price % prev price from bars;
  
  / Average volatility by time of day
  select avg_vol:sqrt[sum ret*ret] by tod from bars
  };

/ ============================================================================
/ TICK-BASED CALCULATIONS
/ ============================================================================

/ Calculate volume-weighted price
vwap:{[prices;sizes]
  (sum prices*sizes)%sum sizes
  };

/ Calculate tick imbalance
tickImbalance:{[tickData]
  / Count upticks vs downticks
  tickData:update priceDelta:deltas price from tickData;
  tickData:update tickDir:signum priceDelta from tickData;
  
  upticks:sum tickData[`tickDir]=1;
  downticks:sum tickData[`tickDir]=-1;
  
  (upticks-downticks)%upticks+downticks
  };

/ ============================================================================
/ MICROSTRUCTURE NOISE
/ ============================================================================

/ Estimate bid-ask spread from high-frequency data
/ Roll (1984) estimator
rollSpread:{[prices]
  / Spread = 2 * sqrt(-cov(delta_p[t], delta_p[t-1]))
  
  diffs:1_deltas prices;
  cov:avg diffs*prev diffs;
  
  2*sqrt neg cov
  };

/ ============================================================================
/ VOLATILITY FORECASTING
/ ============================================================================

/ GARCH(1,1) variance forecast (simplified)
garchForecast:{[returns;omega;alpha;beta;horizon]
  / omega: constant term
  / alpha: ARCH coefficient
  / beta: GARCH coefficient
  / horizon: forecast periods
  
  / Unconditional variance
  uncond_var:omega%(1-alpha-beta);
  
  / Current variance (sample variance)
  curr_var:var returns;
  
  / Multi-step forecast
  forecasts:(horizon#0f);
  forecasts[0]:omega + alpha*last[returns]*last[returns] + beta*curr_var;
  
  i:1;
  while[i<horizon;
    forecasts[i]:omega + (alpha+beta)*forecasts[i-1];
    i+:1
  ];
  
  sqrt forecasts
  };

/ ============================================================================
/ EXPONENTIALLY WEIGHTED MOVING AVERAGE (EWMA)
/ ============================================================================

ewmaVol:{[returns;lambda]
  / lambda: decay factor (e.g., 0.94 for RiskMetrics)
  
  n:count returns;
  weights:(1-lambda)*lambda xexp reverse til n;
  weights:weights%sum weights;
  
  sqrt sum weights*returns*returns
  };

/ ============================================================================
/ AGGREGATION UTILITIES
/ ============================================================================

/ Aggregate ticks to OHLCV bars
ticksToOHLCV:{[tickData;interval]
  / tickData: table with time, sym, price, size
  / interval: time interval
  
  select 
    open:first price,
    high:max price,
    low:min price,
    close:last price,
    volume:sum size,
    vwap:size wavg price,
    count:count i
  by sym, interval xbar time
  from tickData
  };

/ Calculate returns from OHLCV
calculateReturns:{[ohlcv;method]
  / method: `close, `log, `overnight
  
  $[method=`close;
      update ret:(close-prev close)%prev close from ohlcv;
    method=`log;
      update ret:log close%prev close from ohlcv;
    method=`overnight;
      update ret:log open%prev close from ohlcv;
    ohlcv
  ]
  };

/ ============================================================================
/ PERFORMANCE UTILITIES
/ ============================================================================

/ Benchmark function execution
benchmark:{[f;args;n]
  / f: function
  / args: arguments
  / n: number of runs
  
  times:(n#0f);
  i:0;
  while[i<n;
    start:.z.p;
    result:f . args;
    elapsed:`.z.p - start;
    times[i]:`long$elapsed%1000000; / Convert to microseconds
    i+:1
  ];
  
  `mean`min`max`std!(avg times;min times;max times;dev times)
  };

/ ============================================================================
/ EXAMPLE USAGE
/ ============================================================================

/ Load sample data and calculate volatility
exampleVolatilityCalc:{[]
  / Generate sample OHLC data
  n:252;
  dates:.z.d+til n;
  
  / Simulate prices (GBM)
  drift:0.0001;
  vol:0.02;
  rets:drift + vol*n?1f;
  prices:100*prds 1+rets;
  
  / Create OHLC
  ohlc:([] 
    date:dates;
    open:prices;
    high:prices*1+abs n?0.01;
    low:prices*1-abs n?0.01;
    close:prices*1+(n?0.02)-0.01
  );
  
  / Calculate different volatility estimates
  results:([] 
    estimator:`closeToClose`parkinson`garmanKlass`rogersSatchell`yangZhang;
    value:(
      dev 1_deltas log ohlc`close;
      parkinsonVol[ohlc`high;ohlc`low];
      garmanKlassVol[ohlc`open;ohlc`high;ohlc`low;ohlc`close];
      rogersSatchellVol[ohlc`open;ohlc`high;ohlc`low;ohlc`close];
      yangZhangVol[ohlc`open;ohlc`high;ohlc`low;ohlc`close]
    )
  );
  
  / Annualize
  update annualized_vol:sqrt[252]*value from results
  };
