from src.strategies import Strategy

class Context:
    def __init__(self, strategy: Strategy = None):
        self._strategy = strategy
    
    def set_strategy(self, strategy: Strategy):
        self._strategy = strategy

    def plot_control_chart(self, data_path: str, *args, **kwargs):
        self._strategy.set_data(data_path, *args, **kwargs)
        self._strategy.process_data()
        self._strategy.plot()
