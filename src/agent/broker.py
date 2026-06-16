from src.agent import config
from src.agent import journal as journal_module


class Broker:
    def __init__(self) -> None:
        self._session_id: str = ""

    def _client(self):
        from alpaca.trading.client import TradingClient
        return TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=True,  # R6: hardcoded, never configurable
        )

    def is_market_open(self) -> bool:
        try:
            clock = self._client().get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def compute_qty(self, price: float, cash: float, mode: str) -> int:
        if price <= 0:
            return 0
        max_pct = (
            config.MAX_POSITION_PCT_CONSERVATIVE if mode == "conservative"
            else config.MAX_POSITION_PCT_NORMAL
        )
        qty = int(cash * max_pct / price)
        return max(0, qty)

    def place_order(self, ticker: str, side: str, qty: int) -> dict:
        if not self.is_market_open():
            journal_module.log_error(
                source="Broker", error=f"Market closed — order for {ticker} rejected",
                ticker=ticker, session_id=self._session_id,
            )
            return {"ok": False, "order_id": None, "status": "market_closed", "reason": "Market is closed"}
        if qty <= 0:
            return {"ok": False, "order_id": None, "status": "zero_qty", "reason": "qty must be > 0"}
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client().submit_order(req)
            return {
                "ok": True,
                "order_id": str(order.id),
                "status": str(order.status),
                "reason": None,
            }
        except Exception as exc:
            err_str = str(exc)
            if "insufficient" in err_str.lower():
                status = "insufficient_funds"
            elif "rate" in err_str.lower():
                status = "rate_limited"
            elif "market" in err_str.lower() and "closed" in err_str.lower():
                status = "market_closed"
            elif "422" in err_str or "400" in err_str:
                status = "api_error_4xx"
            else:
                status = "broker_unavailable"
            journal_module.log_error(
                source="Broker", error=err_str, ticker=ticker,
                session_id=self._session_id,
            )
            return {"ok": False, "order_id": None, "status": status, "reason": err_str}

    def get_open_orders(self) -> list[dict]:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            orders = self._client().get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            return [
                {"id": str(o.id), "symbol": o.symbol, "side": str(o.side), "qty": str(o.qty)}
                for o in orders
            ]
        except Exception:
            return []

    def cancel_all_orders(self) -> bool:
        try:
            self._client().cancel_orders()
            return True
        except Exception:
            return False
