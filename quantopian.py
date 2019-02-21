#quantopian practice
import quantopian.algorithm as algo
import quantopian.optimize as opt

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline


def initialize(context):
    context.max_leverage = 1.0
    context.max_pos_size = 0.015
    context.max_turnover = 0.95
    
    algo.attach_pipeline(
        make_pipeline(), 'data_pipe')
    
    algo.attach_pipeline(
        risk_loading_pipeline(), 'risk_pipe'
    )
    
    algo.schedule_function(
        rebalance,
        date_rule = algo.date_rules.week_start(),
        time_rule = algo.time_rules.market_open()
    )

def before_trading_start(context, data):
    context.pipeline_data = algo.pipeline_output('data_pipe')
    context.risk_factor_betas = algo.pipeline_output('risk_pipe')
    
def make_pipeline():
    sentiment_score = SimpleMovingAverage(
        inputs = [stocktwits.bull_minus_bear],
        window_length = 3,
        mask = QTradableStocksUS()
    )
    
    return Pipeline(
        columns = {'sentiment_score':sentiment_score}, 
        screen = sentiment_score.notnull()
    )
  
def rebalance(context, data):
    alpha = context.pipeline_data.sentiment_score
    
    if not alpha.empty:
        objective = opt.MaximizeAlpha(alpha)
        
        constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
            -context.max_pos_size,
             constrain.max_pos_size
        )
        
        max_leverage = opt.MaxGrossExposure(context.max_leverage)
        dollar_neutral = opt.DollarNeutral()
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constrains = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            version = opt.Newest
        )
        
        algo.order_optimal_portfolio(
            objective = objective,
            constrains = [
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                max_turnover,
                factor_risk_constrains
            ]
        )
        
        
        
        






'''
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.experiment import QTradableStocksUS
from quantopian.resaerch import prices
import alphalens as al


def make_pipeline():
    
    base_universe = QTradableStocksUS()
    
    sentiment_score = SimpleMovingAverage(
        inputs = [stocktwits.bull_minus_bear],
        window_length = 3)
        
    top_bottom_scores = (
        sentiment_score.top(350)|sentiment_score.bottom(350))
    
    return Pipeline(
        columns = {'sentiment_score': sentiment_score}, 
        screen=(base_universe & top_bottom_scores))

asset_list = pipeline_output.index.levels[1].unique()
asset_prices = prices(
    asset_list,
    start = period_start,
    end = period_end)
    
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor = pipeline_output['sentiment_score'],
    prices = asset_prices,
    quantiles = 2,
    periods = (1, 5, 10))
    
factor_data.head(5)

mean_return_by_q, std_err_by_q = al.performance.mean_return_by_quantile(factor_data)

al.plotting.plot_quantitle_returns_bar(
    mean_return_by_q.apply(
        al.utils.rate_of_return,
        axis = 0,
        args = ('1D')))
        
ls_factor_returns = al.performance.factor_returns(factor_data)
al.plotting.plot_cumulative_returns(ls_factor_returns['5D'], '5D')



