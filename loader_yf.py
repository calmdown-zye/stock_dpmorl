import os
import time
import random
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yahooquery import Ticker
import pandas_datareader.data as web

# --- 로거 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 상수 설정 ---
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 도우미 함수 ---

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 무한대 값을 NaN으로 바꾸고, 모든 값이 NaN인 칼럼을 제거합니다."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how='all')
    return df

def _tag(tickers: list[str], years: int) -> str:
    """캐시 파일명을 위한 고유 태그를 생성합니다."""
    return f"{'-'.join(sorted(tickers))}_{years}y"

def _save_cache(df: pd.DataFrame, tag: str):
    """DataFrame을 Parquet 형식으로 캐시에 저장합니다."""
    path = os.path.join(CACHE_DIR, f"close_{tag}.parquet")
    try:
        df.to_parquet(path)
        logging.info(f"✅ Cache saved successfully to {path}")
    except Exception as e:
        logging.error(f"Failed to save cache to {path}: {e}")

def _load_cache(tag: str) -> pd.DataFrame:
    """캐시에서 Parquet 파일을 읽어 DataFrame으로 반환합니다."""
    path = os.path.join(CACHE_DIR, f"close_{tag}.parquet")
    if os.path.exists(path):
        try:
            logging.info(f"💾 Loading data from cache: {path}")
            return pd.read_parquet(path)
        except Exception as e:
            logging.warning(f"Could not read cache file {path}, attempting to re-download. Error: {e}")
    return pd.DataFrame()

# --- 데이터 소스별 다운로더 함수 ---

def _yf_download_chunked(tickers: list[str], start: str, end: str, **kwargs) -> pd.DataFrame:
    """yfinance를 통해 데이터를 분할 다운로드합니다. (HTTP 429 오류 완화)"""
    chunk_size = kwargs.get('chunk_size', 5)
    pause = kwargs.get('pause', 1.5)
    max_retry = kwargs.get('max_retry', 3)
    
    all_closes = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        for attempt in range(max_retry):
            try:
                session = requests.Session()
                session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
                
                raw = yf.download(
                    chunk, start=start, end=end,
                    auto_adjust=True, progress=False, threads=False, session=session
                )
                
                if raw.empty:
                    logging.warning(f"No data for tickers {chunk} in this period.")
                    break # 성공적으로 비어있는 데이터를 받았으므로 재시도 불필요

                # yfinance는 티커가 하나일 때와 여러 개일 때 다른 구조의 DataFrame을 반환
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw.get('Close', pd.DataFrame())
                else:
                    close = raw[['Close']].rename(columns={'Close': chunk[0]}) if 'Close' in raw else pd.DataFrame()
                
                all_closes.append(close)
                time.sleep(pause)
                break # 청크 성공, 다음 청크로 이동
            except Exception as e:
                logging.warning(f"Attempt {attempt+1}/{max_retry} failed for chunk {chunk}: {e}")
                if attempt + 1 == max_retry:
                    logging.error(f"🚨 Failed to download chunk {chunk} after {max_retry} retries. Aborting yfinance download.")
                    return pd.DataFrame() # 한 청크라도 실패하면 전체 다운로드 실패 처리
                time.sleep(pause * (attempt + 2)) # 점진적 백오프

    if not all_closes:
        return pd.DataFrame()
        
    final_df = pd.concat(all_closes, axis=1)
    return _clean(final_df).sort_index()

def _yahooquery_download(tickers: list[str], years: int, **kwargs) -> pd.DataFrame:
    """yahooquery를 통해 데이터를 다운로드합니다."""
    try:
        t = Ticker(tickers, asynchronous=True, formatted=True)
        hist = t.history(period=f"{years}y")
        
        if not isinstance(hist, dict): # yahooquery v2.3+
            close = hist.get('close', pd.DataFrame()).unstack(level=0)
        else: # 이전 버전 호환성
             close = pd.DataFrame({k: v['close'] for k, v in hist.items() if 'close' in v})
        
        return _clean(close).sort_index()
    except Exception as e:
        logging.error(f"yahooquery download failed: {e}")
        return pd.DataFrame()

def _stooq_download(tickers: list[str], start: str, end: str, **kwargs) -> pd.DataFrame:
    """stooq를 통해 미국 주식 데이터를 다운로드합니다."""
    all_closes = []
    for ticker in tickers:
        try:
            # Stooq는 미국 주식에 .US 접미사를 붙여야 할 수 있음
            symbol = f"{ticker.replace('-', '.')}.US"
            s = web.DataReader(symbol, "stooq", start, end)["Close"].rename(ticker)
            all_closes.append(s)
        except Exception:
            logging.warning(f"Could not download {ticker} from stooq.")
    
    if not all_closes:
        return pd.DataFrame()

    final_df = pd.concat(all_closes, axis=1)
    return _clean(final_df).sort_index()

# --- 메인 함수 ---

def download_close(tickers: list[str], years: int = 10, use_cache: bool = True) -> pd.DataFrame:
    """
    지정된 티커 목록과 기간에 대한 종가 데이터를 다운로드합니다.
    여러 데이터 소스를 순차적으로 시도하며, 로컬 캐시를 활용합니다.

    Args:
        tickers (list[str]): 다운로드할 주식 티커 리스트.
        years (int): 다운로드할 데이터의 기간 (년).
        use_cache (bool): 로컬 캐시 사용 여부.

    Returns:
        pd.DataFrame: 날짜를 인덱스로, 티커를 칼럼으로 하는 종가 데이터.
                       데이터를 가져오지 못하면 빈 DataFrame을 반환.
    """
    tag = _tag(tickers, years)
    if use_cache:
        cached_df = _load_cache(tag)
        if not cached_df.empty:
            return cached_df

    logging.info(f"No cache found for tag '{tag}'. Starting download...")
    
    end = datetime.today()
    start = end - timedelta(days=365.25 * years)
    start_str, end_str = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    
    # 시도할 데이터 소스와 필요한 인자들을 순서대로 정의
    data_sources = [
        ("yfinance", _yf_download_chunked, {"start": start_str, "end": end_str}),
        ("yahooquery", _yahooquery_download, {"years": years}),
        ("stooq", _stooq_download, {"start": start_str, "end": end_str}),
    ]

    final_df = pd.DataFrame()
    for name, downloader, params in data_sources:
        logging.info(f"--- Attempting to download from {name.upper()} ---")
        final_df = downloader(tickers=tickers, **params)
        
        if not final_df.empty:
            logging.info(f"✅ Successfully downloaded data from {name.upper()}.")
            if use_cache:
                _save_cache(final_df, tag)
            return final_df
        else:
            logging.warning(f"⚠️ Failed to get data from {name.upper()}. Trying next source...")
            
    logging.error("🚨 All data sources failed and no cache available.")
    return pd.DataFrame()

# src/data/loader_yf.py
def load_market_frames(tickers, years=3, use_cache=True):
    close = download_close(tickers, years=years, use_cache=use_cache)

    # 1) 날짜 정렬 + dtype
    close = close.sort_index().astype("float32")

    # 2) 완전결측 칼럼 제거 + 앞/뒤 채움
    close = close.dropna(axis=1, how="all").ffill().bfill()

    # 3) 수익률 + 분산 0 자산 제거
    ret = close.pct_change().dropna()
    keep = ret.std() > 0
    close = close.loc[ret.index, keep]
    ret   = ret.loc[:, keep]

    assert close.shape[1] > 0 and ret.shape[1] > 0, "자산이 0개가 됨 – 데이터 로딩/전처리 확인"
    return close, ret

def load_yf_panel(tickers, start=None, end=None, years=10, win_vol=20):
    """
    기존 코드 호환용: (close, ret, vol) 반환.
    start/end로 슬라이스 후 인덱스 정렬 맞춰서 리턴.
    """
    close, ret = load_market_frames(tickers, years=years, use_cache=True)

    if start or end:
        close = close.loc[start:end]
        # 슬라이스 이후 ret 재계산(인덱스 정합 보장)
        ret = close.pct_change().dropna()
        close = close.loc[ret.index]

    vol = ret.rolling(win_vol).std().bfill()
    return close, ret, vol

def compute_base_signals(close: pd.DataFrame, ret: pd.DataFrame, win_mom: int = 20, win_vol: int = 20):
    """
    기존 import 경로 호환용. (mom, val, vol) 시그널 dict 반환.
    - mom: 모멘텀 (win_mom 수익률)
    - val: 밸류 proxy (1/price)
    - vol: 변동성 (ret std rolling win_vol)
    인덱스/컬럼 ret에 맞춰 정렬.
    """
    # 모멘텀: win_mom 기간 종가 변화율
    mom = close.pct_change(win_mom).dropna()

    # 밸류: 1/Close (inf/NaN 정리)
    val = (1.0 / close).replace([np.inf, -np.inf], np.nan).ffill()

    # 변동성: 수익률 std 롤링
    vol = ret.rolling(win_vol).std().bfill()

    # 인덱스/컬럼 정합(환경은 ret 기준으로 굴러가니까 ret.index로 맞춤)
    common_idx = ret.index.intersection(mom.index).intersection(val.index).intersection(vol.index)
    mom = mom.loc[common_idx, ret.columns].astype("float32")
    val = val.loc[common_idx, ret.columns].astype("float32")
    vol = vol.loc[common_idx, ret.columns].astype("float32")

    return {"mom": mom, "val": val, "vol": vol}


if __name__ == '__main__':
    # --- 테스트 예제 ---
    test_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
    
    print("--- 1. 캐시 없이 첫 다운로드 시도 ---")
    data = download_close(tickers=test_tickers, years=1, use_cache=True)
    
    if not data.empty:
        print("\n--- 다운로드 성공! 데이터 확인 ---")
        print(data.head())
        print(data.tail())
        
        print("\n--- 2. 캐시를 사용해 다시 로드 시도 ---")
        # 캐시에서 바로 로드되어야 하므로 매우 빨라야 함
        cached_data = download_close(tickers=test_tickers, years=1, use_cache=True)
        print("로드된 데이터:")
        print(cached_data.head())
    else:
        print("\n--- 데이터 다운로드 실패 ---")