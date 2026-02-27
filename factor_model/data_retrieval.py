from __future__ import annotations

from typing import Optional
import pandas as pd

from .config import SnowflakeConfig, load_snowflake_config

try:
    import snowflake.connector
except Exception as e:  # pragma: no cover
    snowflake = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class SnowflakeDataRetriever:
    """Read-only Snowflake retriever (reader account friendly)."""

    DB_DEFAULT = "EDS_DEV_BERKELEY"
    SCHEMA_DEFAULT = "BERKELEY"

    FUNDAMENTALS_TABLE = "EDS_FACTORS_FUNDAMENTALS_NTM_LTM"
    UNIVERSE_TABLE = "EDS_FACTORS_UNIVERSE"
    PRICES_TABLE = "SPLIT_ADJUSTED_PRICES_EDS_FACTORS"
    ENTERPRISE_VALUE_VIEW = "ENTERPRISEVALUE_HISTORY"
    EXCHANGE_RATES_VIEW = "EXCHANGERATES"
    MARKET_VALUE_VIEW = "MARKET_VALUE_HISTORY"

    def __init__(
        self,
        sf: Optional[SnowflakeConfig] = None,
        env_path: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "snowflake-connector-python is required for Snowflake access. "
                "Install it with: pip install snowflake-connector-python"
            ) from _IMPORT_ERROR

        self.sf = sf or load_snowflake_config(env_path=env_path)
        self.database = database or (self.sf.database or self.DB_DEFAULT)
        self.schema = schema or (self.sf.schema or self.SCHEMA_DEFAULT)

        missing = [k for k in ["account", "user", "password"] if not getattr(self.sf, k)]
        if missing:
            raise ValueError(
                f"Missing Snowflake env vars: {', '.join('SNOWFLAKE_'+m.upper() for m in missing)}. "
                "Put them in your .env (and do NOT commit it)."
            )

    def _connect(self):
        return snowflake.connector.connect(
            account=self.sf.account,
            user=self.sf.user,
            password=self.sf.password,
            warehouse=self.sf.warehouse,
            role=self.sf.role,
            database=self.database,
            schema=self.schema,
        )

    def _fq(self, obj: str) -> str:
        return f"{self.database}.{self.schema}.{obj}"

    def query(self, sql: str) -> pd.DataFrame:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql)
                try:
                    df = cur.fetch_pandas_all()
                except Exception:
                    df = pd.read_sql(sql, conn)
            finally:
                cur.close()
        return df

    def get_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        cols = ["DATE","FACTSET_ID","ADJUSTED_PRICE","ADJUSTED_VOLUME","IS_HOLIDAY"]
        sql = f"""
        SELECT {', '.join(cols)}
        FROM {self._fq(self.PRICES_TABLE)}
        WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
        """
        df = self.query(sql)
        df["DATE"] = pd.to_datetime(df["DATE"])
        if "IS_HOLIDAY" in df.columns:
            df = df[df["IS_HOLIDAY"] == False].copy()
            df.drop(columns=["IS_HOLIDAY"], inplace=True)
        return df

    def get_fundamentals(self, start_date: str, end_date: str) -> pd.DataFrame:
        cols = [
            "DATE","FACTSET_ID",
            "COGS_LTM","COGS_NTM",
            "EBITDA_LTM","EBITDA_NTM",
            "EPS_LTM","EPS_NTM",
            "SALES_LTM","SALES_NTM",
        ]
        sql = f"""
        SELECT {', '.join(cols)}
        FROM {self._fq(self.FUNDAMENTALS_TABLE)}
        WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
        """
        df = self.query(sql)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    def get_universe(self) -> pd.DataFrame:
        cols = ["BLOOMBERG_TICKER","COUNTRY","FACTSET_ID","INDUSTRY","INDUSTRYGROUP","NAME","SECTOR"]
        sql = f"""
        SELECT {', '.join(cols)}
        FROM {self._fq(self.UNIVERSE_TABLE)}
        """
        return self.query(sql)

    # Optional objects you listed
    def get_enterprise_value_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        cols = ["DATE","ENTERPRISE_VALUE","EV_COMPONENTS_DATE","FACTSET_ID"]
        sql = f"""
        SELECT {', '.join(cols)}
        FROM {self._fq(self.ENTERPRISE_VALUE_VIEW)}
        WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
        """
        df = self.query(sql)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    def get_exchange_rates(self, start_date: str, end_date: str) -> pd.DataFrame:
        cols = ["CURRENCYCODE","DATE","EXCHANGERATE","LAG_EXCHANGERATE"]
        sql = f"""
        SELECT {', '.join(cols)}
        FROM {self._fq(self.EXCHANGE_RATES_VIEW)}
        WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
        """
        df = self.query(sql)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    def get_market_value_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        cols = ["CURRENCY","DATE","FACTSET_ID","MARKETCAP"]
        sql = f"""
        SELECT {', '.join(cols)}
        FROM {self._fq(self.MARKET_VALUE_VIEW)}
        WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
        """
        df = self.query(sql)
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    @staticmethod
    def calculate_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame(columns=["DATE","FACTSET_ID","RETURN"])
        df = prices.sort_values(["FACTSET_ID","DATE"]).copy()
        df["RETURN"] = df.groupby("FACTSET_ID", sort=False)["ADJUSTED_PRICE"].pct_change()
        return df[["DATE","FACTSET_ID","RETURN"]]
