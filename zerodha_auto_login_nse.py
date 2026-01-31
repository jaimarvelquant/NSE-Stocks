import json
import os
import traceback
from datetime import date, timedelta, datetime, time as dt_time
import requests
import pyotp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kiteconnect import KiteConnect
from logzero import logger, logfile
from threading import Event, Thread
import sys
import time
import re
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import logging
from pathlib import Path
import shutil
import subprocess
from collections import defaultdict

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Hardcoded credentials
CREDENTIALS = {
    "user_id": "ZM9343",
    "password": "Chetti@23456",
    "totp_key": "WYFGJCB4XNSJNAPTGXI26BGO4A2PTGHF",
    "api_key": "90mo8gkxxg2p0wm0",
    "api_secret": "hy7impfdog70ctuwkx76fuo8h5dincmb"
}

# Telegram alert settings
TELEGRAM_BOT_TOKEN = "8471418378:AAERXNyXK_SLuAWETfVLInuxIodnQ16MMGU"
TELEGRAM_CHAT_ID = "-1003599016958"
TELEGRAM_BOT_NAME = "@ajai_fullalert_bot"


def send_telegram_alert(message: str) -> None:
    """Send a Telegram message; log warning if it fails but don't break flow."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }
        # Use a short timeout so alerts never block the main flow
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.warning(f"Failed to send Telegram alert: {e}")

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global variables
today = datetime.now().strftime("%Y-%m-%d")
logfile(filename=os.path.join(script_dir, f'logfile_auto_login_{today}.log'))
kite = KiteConnect(CREDENTIALS["api_key"])

# Global variable to store request_token from callback
callback_request_token = None
callback_event = Event()

class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP server handler to catch the OAuth callback"""
    def do_GET(self):
        global callback_request_token
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        if 'request_token' in query_params:
            callback_request_token = query_params['request_token'][0]
            logger.info(f"Received request_token from callback: {callback_request_token[:20]}...")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>')
            
            callback_event.set()
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Error: No request_token found</h1></body></html>')
            logger.warning(f"Callback received but no request_token found. Path: {self.path}")
    
    def log_message(self, format, *args):
        pass

def start_callback_server(port=5001, timeout=120):
    """Start HTTP server to catch OAuth callback"""
    global callback_request_token, callback_event
    
    callback_request_token = None
    callback_event.clear()
    
    server = HTTPServer(('127.0.0.1', port), CallbackHandler)
    logger.info(f"Started callback server on http://127.0.0.1:{port}")
    
    server.timeout = 1.0
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        server.handle_request()
        if callback_event.is_set():
            logger.info("Callback received, shutting down server...")
            break
        time.sleep(0.1)
    
    server.server_close()
    
    if callback_request_token:
        return callback_request_token
    else:
        logger.error(f"Callback server timed out after {timeout} seconds")
        return None

def update_env_file(api_key, access_token):
    """Update or create .env file with access token for fetch_historical_data_cu.py"""
    env_path = os.path.join(script_dir, ".env")
    
    env_content = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_content[key.strip()] = value.strip()
    
    env_content['ZERODHA_API_KEY'] = api_key
    env_content['ZERODHA_ACCESS_TOKEN'] = access_token
    
    with open(env_path, 'w') as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"Updated .env file with new access token at {env_path}")

def auto_login(user_id, password, totp_key, api_key, api_secret):
    try:
        req_session = requests.Session()
        login_url = "https://kite.zerodha.com/api/login"
        two_fa_url = "https://kite.zerodha.com/api/twofa"
        twofa = f"{pyotp.TOTP(totp_key).now()}"
        
        logger.info(f"Attempting login for user: {user_id}")
        login_response = req_session.post(
            login_url,
            data={
                "user_id": user_id,
                "password": password,
            },
            verify=False
        )
        
        if login_response.status_code != 200:
            logger.error(f"Login failed with status code: {login_response.status_code}")
            logger.error(f"Response: {login_response.text}")
            return None
        
        login_data = login_response.json()
        if "data" not in login_data or "request_id" not in login_data["data"]:
            logger.error(f"Login response missing request_id. Response: {login_data}")
            return None
            
        request_id = login_data["data"]["request_id"]
        logger.info(f"Login successful. Request ID: {request_id}")
        
        # Retry 2FA with exponential backoff
        twofa_response = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                twofa_response = req_session.post(
                    two_fa_url,
                    data={"user_id": user_id, "request_id": request_id, "twofa_value": twofa},
                    verify=False,
                    timeout=30
                )
                if twofa_response.status_code == 200:
                    break
                else:
                    logger.warning(f"2FA attempt {retry + 1} failed with status code: {twofa_response.status_code}")
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 2  # 2, 4, 6 seconds
                        logger.info(f"Retrying 2FA in {wait_time} seconds...")
                        time.sleep(wait_time)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                logger.warning(f"2FA attempt {retry + 1} failed with connection error: {e}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # 2, 4, 6 seconds
                    logger.info(f"Retrying 2FA in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"2FA failed after {max_retries} attempts due to connection error")
                    return None
        
        if twofa_response is None or twofa_response.status_code != 200:
            logger.error(f"2FA failed with status code: {twofa_response.status_code if twofa_response else 'None'}")
            if twofa_response:
                logger.error(f"Response: {twofa_response.text}")
            return None
            
        logger.info("2FA completed successfully")
        
        connect_url = f"https://kite.trade/connect/login?api_key={api_key}"
        logger.info(f"Requesting connection URL: {connect_url}")
        
        api_session = None
        try:
            api_session = req_session.get(connect_url, verify=False, allow_redirects=True, timeout=30)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(f"Connection to callback URL failed (expected if no server on port 5001): {e}")
            logger.info("Attempting to extract request_token from redirect headers...")
            
            try:
                manual_response = req_session.get(connect_url, verify=False, allow_redirects=False, timeout=30)
                redirect_url = None
                
                max_redirects = 10
                current_response = manual_response
                for _ in range(max_redirects):
                    if current_response.status_code in [301, 302, 303, 307, 308]:
                        if 'Location' in current_response.headers:
                            redirect_url = current_response.headers['Location']
                            logger.info(f"Found redirect to: {redirect_url}")
                            
                            if "request_token=" in redirect_url:
                                break
                            
                            if redirect_url.startswith('http'):
                                current_response = req_session.get(redirect_url, verify=False, allow_redirects=False, timeout=30)
                            else:
                                break
                        else:
                            break
                    else:
                        break
                
                if redirect_url and "request_token=" in redirect_url:
                    url_parts = redirect_url.split("request_token=")
                    if len(url_parts) >= 2:
                        request_token = url_parts[1].split("&")[0].split("#")[0]
                        logger.info(f"Request token extracted from redirect URL: {request_token[:20]}...")
                        data = kite.generate_session(request_token, api_secret=api_secret)
                        access_token = data["access_token"]
                        logger.info("Session generated successfully. Access token obtained.")
                        return request_token, access_token, kite
                
            except Exception as manual_error:
                logger.error(f"Failed to extract redirect manually: {manual_error}")
            
            logger.error("Unable to extract request_token. Please check:")
            logger.error("1. The redirect URL in app settings: http://127.0.0.1:5001/zerodha/callback")
            logger.error("2. That the API key is active and properly configured")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during redirect: {e}")
            logger.error(traceback.format_exc())
            return None
        
        if api_session.history:
            logger.info(f"Redirect chain ({len(api_session.history)} redirects):")
            for i, redirect in enumerate(api_session.history, 1):
                logger.info(f"  Redirect {i}: {redirect.url}")
                if 'Location' in redirect.headers:
                    logger.info(f"    Location header: {redirect.headers['Location']}")
        
        final_url = api_session.url
        logger.info(f"Final redirect URL: {final_url}")
        
        if "/connect/authorize" in final_url:
            logger.info("Reached authorize page. Attempting to complete authorization...")
            
            sess_id = None
            if "sess_id=" in final_url:
                sess_id = final_url.split("sess_id=")[1].split("&")[0]
                logger.info(f"Extracted sess_id: {sess_id}")
            
            try:
                logger.info("Checking authorize page Location header...")
                auth_no_redirect = req_session.get(final_url, verify=False, allow_redirects=False, timeout=30)
                
                if auth_no_redirect.status_code in [301, 302, 303, 307, 308] and 'Location' in auth_no_redirect.headers:
                    location = auth_no_redirect.headers['Location']
                    logger.info(f"Found Location header in authorize response: {location}")
                    if "request_token=" in location:
                        url_parts = location.split("request_token=")
                        if len(url_parts) >= 2:
                            request_token = url_parts[1].split("&")[0].split("#")[0]
                            logger.info(f"Request token found in Location header: {request_token[:20]}...")
                            data = kite.generate_session(request_token, api_secret=api_secret)
                            access_token = data["access_token"]
                            logger.info("Session generated successfully. Access token obtained.")
                            return request_token, access_token, kite
                
                logger.info("Trying authorize page with redirects enabled...")
                authorize_response = req_session.get(final_url, verify=False, allow_redirects=True, timeout=30)
                
                if authorize_response.history:
                    logger.info(f"Authorization triggered redirects ({len(authorize_response.history)} redirects):")
                    for i, redirect in enumerate(authorize_response.history, 1):
                        logger.info(f"  Auth redirect {i}: {redirect.url}")
                        if 'Location' in redirect.headers:
                            logger.info(f"    Location: {redirect.headers['Location']}")
                
                final_auth_url = authorize_response.url
                logger.info(f"Final URL after authorization: {final_auth_url}")
                
                if "request_token=" in final_auth_url:
                    url_parts = final_auth_url.split("request_token=")
                    if len(url_parts) >= 2:
                        request_token = url_parts[1].split("&")[0].split("#")[0]
                        logger.info(f"Request token found after authorization: {request_token[:20]}...")
                        data = kite.generate_session(request_token, api_secret=api_secret)
                        access_token = data["access_token"]
                        logger.info("Session generated successfully. Access token obtained.")
                        return request_token, access_token, kite
                
                if authorize_response.history:
                    for redirect in authorize_response.history:
                        if 'Location' in redirect.headers:
                            location = redirect.headers['Location']
                            if "request_token=" in location:
                                url_parts = location.split("request_token=")
                                if len(url_parts) >= 2:
                                    request_token = url_parts[1].split("&")[0].split("#")[0]
                                    logger.info(f"Request token found in authorization redirect: {request_token[:20]}...")
                                    data = kite.generate_session(request_token, api_secret=api_secret)
                                    access_token = data["access_token"]
                                    logger.info("Session generated successfully. Access token obtained.")
                                    return request_token, access_token, kite
                
                if "sess_id=" in final_url:
                    sess_id = final_url.split("sess_id=")[1].split("&")[0]
                    logger.info(f"Extracted sess_id: {sess_id}")
                    
                    try:
                        authorize_post_url = f"https://kite.zerodha.com/connect/authorize"
                        post_data = {
                            "sess_id": sess_id,
                            "api_key": api_key
                        }
                        logger.info(f"Attempting POST to authorize endpoint...")
                        post_response = req_session.post(authorize_post_url, data=post_data, verify=False, allow_redirects=True, timeout=30)
                        
                        if post_response.history:
                            logger.info(f"POST authorization triggered redirects:")
                            for redirect in post_response.history:
                                logger.info(f"  Redirect: {redirect.url}")
                                if 'Location' in redirect.headers:
                                    location = redirect.headers['Location']
                                    logger.info(f"    Location: {location}")
                                    if "request_token=" in location:
                                        url_parts = location.split("request_token=")
                                        if len(url_parts) >= 2:
                                            request_token = url_parts[1].split("&")[0].split("#")[0]
                                            logger.info(f"Request token found in POST redirect: {request_token[:20]}...")
                                            data = kite.generate_session(request_token, api_secret=api_secret)
                                            access_token = data["access_token"]
                                            logger.info("Session generated successfully. Access token obtained.")
                                            return request_token, access_token, kite
                        
                        final_post_url = post_response.url
                        if "request_token=" in final_post_url:
                            url_parts = final_post_url.split("request_token=")
                            if len(url_parts) >= 2:
                                request_token = url_parts[1].split("&")[0].split("#")[0]
                                logger.info(f"Request token found in POST final URL: {request_token[:20]}...")
                                data = kite.generate_session(request_token, api_secret=api_secret)
                                access_token = data["access_token"]
                                logger.info("Session generated successfully. Access token obtained.")
                                return request_token, access_token, kite
                                
                    except Exception as post_error:
                        logger.warning(f"POST authorization attempt failed: {post_error}")
                
                logger.info("Programmatic authorization failed. Opening browser for manual authorization...")
                logger.info(f"Authorization URL: {final_url}")
                logger.info("IMPORTANT: You need to login and authorize in the browser using the SAME session.")
                logger.info("Waiting for callback on http://127.0.0.1:5001/zerodha/callback...")
                
                callback_thread = Thread(target=start_callback_server, args=(5001, 120), daemon=True)
                callback_thread.start()
                
                time.sleep(1)
                
                try:
                    webbrowser.open(final_url)
                    logger.info("Browser opened. Please complete authorization...")
                except Exception as browser_error:
                    logger.error(f"Failed to open browser: {browser_error}")
                    logger.info(f"Please manually open this URL in your browser: {final_url}")
                
                logger.info("Waiting for authorization callback (max 120 seconds)...")
                if callback_event.wait(timeout=120):
                    if callback_request_token:
                        logger.info(f"Request token received from callback: {callback_request_token[:20]}...")
                        data = kite.generate_session(callback_request_token, api_secret=api_secret)
                        access_token = data["access_token"]
                        logger.info("Session generated successfully. Access token obtained.")
                        return callback_request_token, access_token, kite
                    else:
                        logger.error("Callback received but no request_token found")
                else:
                    logger.error("Timeout waiting for authorization callback")
                
                return None
                
            except Exception as auth_error:
                logger.error(f"Error during authorization: {auth_error}")
        
        request_token = None
        urls_to_check = [final_url]
        
        if api_session.history:
            urls_to_check.extend([r.url for r in api_session.history])
            for redirect in api_session.history:
                if 'Location' in redirect.headers:
                    urls_to_check.append(redirect.headers['Location'])
        
        if 'Location' in api_session.headers:
            urls_to_check.append(api_session.headers['Location'])
        
        seen = set()
        unique_urls = []
        for url in urls_to_check:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        for url in unique_urls:
            if "request_token=" in url:
                url_parts = url.split("request_token=")
                if len(url_parts) >= 2:
                    request_token = url_parts[1].split("&")[0].split("#")[0]
                    logger.info(f"Request token found in URL: {url}")
                    logger.info(f"Request token extracted: {request_token[:20]}...")
                    break
        
        if not request_token:
            logger.error(f"ERROR: request_token not found in any redirect URL!")
            logger.error(f"Checked URLs:")
            for url in urls_to_check:
                logger.error(f"  - {url}")
            logger.error(f"This usually means:")
            logger.error(f"1. The redirect URL is not configured correctly in your Kite Connect app settings")
            logger.error(f"2. The redirect URL in app settings doesn't match: http://127.0.0.1:5001/zerodha/callback")
            logger.error(f"3. The API key might not be properly activated")
            return None
        
        logger.info("Generating session with request_token...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        logger.info("Session generated successfully. Access token obtained.")
        
        return request_token, access_token, kite
    except Exception as e:
        logger.error(f"Error in auto_login: {str(e)}")
        logger.error(traceback.format_exc())
        return None

class ZerodhaDataFetcher:
    """Class to fetch 1-minute historical data for NSE stocks"""
    
    _INPUT_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]

    def __init__(self, kite_instance):
        self.kite = kite_instance
        self.instruments_df = self.get_instruments()
        logger.info(f"Loaded {len(self.instruments_df)} instruments")

    def get_instruments(self):
        """Fetch all available instruments from Zerodha"""
        try:
            instruments = self.kite.instruments()
            df = pd.DataFrame(instruments)
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
            df = df[df['exchange'].isin(['NSE', 'NFO'])]
            return df
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return pd.DataFrame()

    def _parse_input_date(self, value, is_end=False):
        """Parse date/datetime or string in fixed formats"""
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            last_err = None
            for fmt in self._INPUT_FORMATS:
                try:
                    dt = datetime.strptime(value.strip(), fmt)
                    break
                except Exception as ex:
                    last_err = ex
                    dt = None
            if dt is None:
                raise ValueError(f"Invalid date format: {value}. Expected one of: {self._INPUT_FORMATS}") from last_err
        else:
            raise ValueError(f"Unsupported date input type: {type(value)}")

        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            if is_end:
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=0)
            else:
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt

    def _normalize_range_from_to(self, from_input, to_input):
        now = datetime.now()
        start_dt = self._parse_input_date(from_input if from_input is not None else now - timedelta(days=1), is_end=False)
        end_dt = self._parse_input_date(to_input if to_input is not None else now, is_end=True)
        if end_dt > now:
            end_dt = now
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        return start_dt, end_dt

    def _transform_dataframe(self, df, stock_name, instrument_type, strike=None, expiry=None):
        """Apply transformations from direct.py: date/time/price/OI conversion"""
        if df.empty:
            return df
        
        df_work = df.copy()
        
        # Handle date column - ensure it's datetime
        if "date" in df_work.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_work["date"]):
                df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
            df_work = df_work.dropna(subset=["date"])
        
        if df_work.empty:
            return df_work
        
        # Convert time to seconds since midnight (integer)
        df_work["time"] = (df_work["date"].dt.hour * 3600 + 
                           df_work["date"].dt.minute * 60 + 
                           df_work["date"].dt.second).astype(int)
        
        # Convert date to yymmdd format as integer
        df_work["date"] = df_work["date"].dt.strftime("%y%m%d").astype(int)
        
        # Convert OHLC to paise (multiply by 100) and ensure integer
        for col in ["open", "high", "low", "close"]:
            if col in df_work.columns:
                try:
                    df_work[col] = (df_work[col] * 100).astype(int)
                except Exception as e:
                    logger.warning(f"Error converting {col} to paise: {e}")
                    # If conversion fails, try to fill with None or 0
                    df_work[col] = 0
        
        # Clean and convert OI if present (handle comma-separated numbers)
        if "oi" in df_work.columns:
            df_work["oi"] = df_work["oi"].astype(str).str.replace(",", "").astype(float).astype(int)
        
        # Create symbol and set columns based on instrument type
        if instrument_type in ["CE", "PE"] and expiry and strike is not None:
            # CE/PE: Create symbol in format {stock}{expiry}{strike}{type}
            # Handle expiry - could be date object, string, or Timestamp
            if isinstance(expiry, date):
                expiry_dt = pd.Timestamp(expiry)
            elif isinstance(expiry, str):
                expiry_dt = pd.to_datetime(expiry)
            else:
                expiry_dt = expiry
            
            expiry_str = expiry_dt.strftime("%d%b%y").upper()  # 30DEC25
            expiry_yymmdd = int(expiry_dt.strftime("%y%m%d"))
            
            # Format strike nicely
            if isinstance(strike, float) and strike.is_integer():
                strike_str = str(int(strike))
            else:
                strike_str = str(strike)
            
            df_work["symbol"] = f"{stock_name}{expiry_str}{strike_str}{instrument_type}"
            df_work["strike"] = strike
            df_work["expiry"] = expiry_yymmdd
            
            # Ensure OI and volume are integers
            if "oi" not in df_work.columns:
                df_work["oi"] = 0
            if "volume" in df_work.columns:
                df_work["volume"] = df_work["volume"].astype(int)
            
            cols = ["date", "time", "symbol", "strike", "expiry", "open", "high", "low", "close", "volume", "oi", "coi"]
            
        elif instrument_type in ["FUT", "FUTURE"]:
            # FUTURE: symbol is just stock name
            df_work["symbol"] = stock_name
            
            # Ensure OI and volume are integers
            if "oi" not in df_work.columns:
                df_work["oi"] = 0
            if "volume" in df_work.columns:
                df_work["volume"] = df_work["volume"].astype(int)
            
            cols = ["date", "time", "symbol", "open", "high", "low", "close", "volume", "oi", "coi"]
            
        else:  # CASH
            df_work["symbol"] = stock_name
            cols = ["date", "time", "symbol", "open", "high", "low", "close"]
        
        # Fill missing columns
        for c in cols:
            if c not in df_work.columns:
                if c == "coi":
                    df_work[c] = 0  # Initialize COI to 0, will be calculated after merging
                else:
                    df_work[c] = None
        
        return df_work[cols]

    def _calculate_coi(self, merged_df, instrument_type):
        """Calculate change in OI (COI) after merging"""
        if instrument_type in ["CE", "PE", "FUT", "FUTURE"] and "oi" in merged_df.columns:
            # Ensure OI is integer
            merged_df["oi"] = merged_df["oi"].astype(int)
            # Sort by symbol, date, and time, then calculate COI per symbol
            merged_df = merged_df.sort_values(by=["symbol", "date", "time"]).reset_index(drop=True)
            merged_df["coi"] = merged_df.groupby("symbol")["oi"].diff().fillna(0).astype(int)
            # Re-sort by date, symbol, time for saving
            merged_df = merged_df.sort_values(by=["date", "symbol", "time"]).reset_index(drop=True)
        elif "coi" in merged_df.columns:
            # For CASH or if COI column exists but shouldn't be calculated, set to 0
            merged_df["coi"] = 0
        
        return merged_df

    def _read_parquet_files_from_date_folders(self, base_folder, folder_type, from_date=None, to_date=None):
        """Read Parquet files from date folders structure: raw_parquet/YEAR/MONTH/YYYYMMDD/TYPE/
        
        Args:
            base_folder: Base folder path (raw_parquet)
            folder_type: Type of folder (CE, PE, FUTURE, CASH)
            from_date: Start date for filtering (date object or datetime)
            to_date: End date for filtering (date object or datetime)
        """
        from glob import glob
        
        folder_map = {
            "CE": ["ce", "CE", "call", "Call", "CALL"],
            "PE": ["pe", "PE", "put", "Put", "PUT"],
            "FUTURE": ["fut", "FUT", "future", "Future", "FUTURE"],
            "CASH": ["cash", "Cash", "CASH"]
        }
        
        possible_names = folder_map.get(folder_type, [folder_type])
        all_files = []
        
        # Convert from_date and to_date to date objects if provided
        target_from_date = None
        target_to_date = None
        if from_date:
            if isinstance(from_date, datetime):
                target_from_date = from_date.date()
            elif isinstance(from_date, str):
                target_from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
            else:
                target_from_date = from_date
        
        if to_date:
            if isinstance(to_date, datetime):
                target_to_date = to_date.date()
            elif isinstance(to_date, str):
                target_to_date = datetime.strptime(to_date, "%Y-%m-%d").date()
            else:
                target_to_date = to_date
        
        try:
            if not os.path.exists(base_folder):
                logger.warning(f"Base folder does not exist: {base_folder}")
                return all_files
            
            # Traverse YEAR folders
            for year_item in os.listdir(base_folder):
                year_path = os.path.join(base_folder, year_item)
                if not os.path.isdir(year_path) or not year_item.isdigit() or len(year_item) != 4:
                    continue
                
                # Traverse MONTH folders
                for month_item in os.listdir(year_path):
                    month_path = os.path.join(year_path, month_item)
                    if not os.path.isdir(month_path) or not month_item.isdigit() or len(month_item) != 2:
                        continue
                    
                    # Traverse YYYYMMDD date folders
                    for date_item in os.listdir(month_path):
                        date_path = os.path.join(month_path, date_item)
                        if not os.path.isdir(date_path) or not date_item.isdigit() or len(date_item) != 8:
                            continue
                        
                        # Extract date from folder name (YYYYMMDD)
                        try:
                            folder_date = datetime.strptime(date_item, "%Y%m%d").date()
                            
                            # Filter by date range if provided
                            if target_from_date and folder_date < target_from_date:
                                continue
                            if target_to_date and folder_date > target_to_date:
                                continue
                        except ValueError:
                            logger.warning(f"Invalid date folder name: {date_item}")
                            continue
                        
                        # Look for TYPE folders inside date folder
                        try:
                            for sub_item in os.listdir(date_path):
                                sub_item_path = os.path.join(date_path, sub_item)
                                if os.path.isdir(sub_item_path) and sub_item.upper() in [name.upper() for name in possible_names]:
                                    parquet_files = glob(os.path.join(sub_item_path, "*.parquet"))
                                    all_files.extend(parquet_files)
                                    if parquet_files:
                                        logger.info(f"    Found {len(parquet_files)} Parquet file(s) in {year_item}/{month_item}/{date_item}/{sub_item}")
                        except Exception as e:
                            logger.warning(f"Error reading folder {date_path}: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error reading base folder {base_folder}: {e}")
        
        return all_files

    def _extract_info_from_filename(self, file, option_type=None):
        """Extract stock name, strike, expiry from filename"""
        base = os.path.basename(file).replace(".parquet", "")
        parts = base.split("_")
        stock = parts[0]
        
        strike = None
        expiry = None
        
        try:
            if option_type in ["CE", "PE"]:
                if "STRIKE" in parts and "EXPIRY" in parts:
                    strike_idx = parts.index("STRIKE") + 1
                    expiry_idx = parts.index("EXPIRY") + 1
                    strike = float(parts[strike_idx])
                    expiry = parts[expiry_idx]
                elif len(parts) >= 4:
                    strike = float(parts[2])
                    expiry = parts[3]
            elif option_type == "FUTURE":
                # Parse expiry from filename: STOCK_future_YYYY-MM-DD_1min.parquet
                if len(parts) >= 3:
                    expiry = parts[2]  # "2025-12-30"
            elif option_type == "CASH":
                strike = None
                expiry = None
        except Exception as e:
            logger.warning(f"Could not parse strike/expiry for {file}: {e}")
        
        return stock, strike, expiry

    def fetch_historical_data(self, symbol, from_date, to_date):
        """Fetch historical 1-minute data for a symbol including cash, futures, and options"""
        try:
            output_base = "raw_parquet"
            os.makedirs(output_base, exist_ok=True)

            # Fetch cash data
            cash_token = self.get_instrument_token(symbol, "EQ")
            if cash_token:
                logger.info(f"Fetching cash data for {symbol}")
                cash_data = self.fetch_1min_data(cash_token, from_date, to_date)
                if not cash_data.empty:
                    self.save_data(cash_data, symbol, "CASH", output_base)

            # Fetch futures data
            futures = self.instruments_df[
                (self.instruments_df['name'] == symbol) & 
                (self.instruments_df['instrument_type'] == 'FUT') &
                (self.instruments_df['expiry'] >= pd.Timestamp(from_date))
            ]
            
            for _, future in futures.iterrows():
                logger.info(f"Fetching futures data for {symbol} expiry {future['expiry'].date()}")
                fut_data = self.fetch_1min_data(future['instrument_token'], from_date, to_date)
                if not fut_data.empty:
                    expiry_date = future['expiry'].date()
                    self.save_data(fut_data, symbol, "FUT", output_base, expiry=expiry_date)
                time.sleep(0.3)

            # Fetch options data
            options = self.instruments_df[
                (self.instruments_df['name'] == symbol) & 
                (self.instruments_df['instrument_type'].isin(['CE', 'PE'])) &
                (self.instruments_df['expiry'] >= pd.Timestamp(from_date))
            ]
            
            for _, option in options.iterrows():
                logger.info(f"Fetching {option['instrument_type']} data for {symbol} strike {option['strike']} expiry {option['expiry'].date()}")
                opt_data = self.fetch_1min_data(option['instrument_token'], from_date, to_date)
                if not opt_data.empty:
                    strike = option['strike']
                    expiry_date = option['expiry'].date()
                    self.save_data(
                        opt_data, 
                        symbol, 
                        option['instrument_type'],
                        output_base,
                        strike=strike,
                        expiry=expiry_date
                    )
                time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")

    def get_instrument_token(self, symbol, instrument_type="EQ"):
        """Get instrument token for a symbol"""
        try:
            instrument = self.instruments_df[
                (self.instruments_df['tradingsymbol'] == symbol) & 
                (self.instruments_df['instrument_type'] == instrument_type)
            ]
            
            if len(instrument) == 0:
                logger.warning(f"No {instrument_type} instrument found for {symbol}")
                return None
                
            return instrument.iloc[0]['instrument_token']
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol}: {e}")
            return None

    def fetch_1min_data(self, instrument_token, from_date, to_date):
        """Fetch 1-minute historical data"""
        try:
            from_date, to_date = self._normalize_range_from_to(from_date, to_date)

            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="minute",
                continuous=False,
                oi=True
            )
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter for market hours (9:15 AM to 3:30 PM)
                market_start = dt_time(9, 15)
                market_end = dt_time(15, 30)
                df = df[
                    (df['date'].dt.time >= market_start) & 
                    (df['date'].dt.time <= market_end)
                ]
                
                # Add strike and expiry columns for options data
                if instrument_token in self.instruments_df['instrument_token'].values:
                    instrument_info = self.instruments_df[
                        self.instruments_df['instrument_token'] == instrument_token
                    ].iloc[0]
                    
                    if instrument_info['instrument_type'] in ['CE', 'PE']:
                        df['strike'] = instrument_info['strike']
                        df['expiry'] = instrument_info['expiry'].date()
                
                return df
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching 1min data: {e}")
            return pd.DataFrame()

    def save_data(self, df, symbol, instrument_type, base_path, strike=None, expiry=None):
        """Save data to CSV file, grouped by date"""
        try:
            if df.empty:
                return

            df_work = df.copy()
            if isinstance(df_work['date'].dtype, pd.DatetimeTZDtype):
                df_work['date'] = df_work['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)

            df_work['trade_date'] = df_work['date'].dt.date

            for trade_date, day_df in df_work.groupby('trade_date'):
                date_str = trade_date.strftime('%Y%m%d')
                year = trade_date.year
                month_num = f"{trade_date.month:02d}"
                type_folder = os.path.join(base_path, str(year), month_num, date_str, instrument_type)
                os.makedirs(type_folder, exist_ok=True)

                if instrument_type == "CASH":
                    filename = f"{symbol}_cash_1min.parquet"
                elif instrument_type == "FUT":
                    filename = f"{symbol}_future_{expiry}_1min.parquet"
                else:
                    strike_str = f"{int(strike)}" if float(strike).is_integer() else f"{strike}"
                    filename = f"{symbol}_{instrument_type}_STRIKE_{strike_str}_EXPIRY_{expiry}_1min.parquet"

                filepath = os.path.join(type_folder, filename)

                df_out = day_df.drop(columns=['trade_date']).copy()
                # Keep datetime as proper dtype; Parquet preserves types
                df_out.to_parquet(filepath, index=False, compression="snappy")
                logger.info(f"Saved {len(df_out)} records to {filepath}")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def save_transformed_data(self, symbol, cash_dataframes, future_dataframes, ce_dataframes, pe_dataframes, base_path):
        """Save transformed and merged data directly to raw_data folder structure"""
        try:
            output_folder = "/home/ubuntu/raw_data"
            
            # Process CASH
            if cash_dataframes:
                self._process_and_save_transformed(
                    symbol, cash_dataframes, "CASH", output_folder, None, None
                )
            
            # Process FUTURE - need to handle nearest expiry per date folder
            if future_dataframes:
                self._process_futures_transformed(symbol, future_dataframes, output_folder)
            
            # Process CE (Call Options)
            if ce_dataframes:
                self._process_options_transformed(symbol, ce_dataframes, "CE", output_folder)
            
            # Process PE (Put Options)
            if pe_dataframes:
                self._process_options_transformed(symbol, pe_dataframes, "PE", output_folder)
                
        except Exception as e:
            logger.error(f"Error saving transformed data for {symbol}: {e}")
            logger.error(traceback.format_exc())

    def _process_and_save_transformed(self, symbol, dataframes, instrument_type, output_folder, strike, expiry):
        """Process and save transformed dataframes for a given instrument type"""
        if not dataframes:
            return
        
        df_list = []
        
        for df in dataframes:
            if df.empty:
                continue
            
            # Apply transformations
            transformed_df = self._transform_dataframe(df, symbol, instrument_type, strike, expiry)
            if not transformed_df.empty:
                df_list.append(transformed_df)
        
        if not df_list:
            return
        
        # Merge all dataframes
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # Calculate COI
        merged_df = self._calculate_coi(merged_df, instrument_type)
        
        # Split by date and save
        for date, df_date in merged_df.groupby("date"):
            merge_date = str(date)  # Convert integer to string for filename (YYMMDD format)
            
            # Extract year, month, and full date from YYMMDD
            # Zero-pad to ensure 6 digits
            date_str = f"{int(date):06d}"  # "251231" or "051231"
            yy = date_str[:2]     # "25" or "05"
            mm = date_str[2:4]    # "12" or "12"
            dd = date_str[4:6]    # "31" or "31"
            year = int(f"20{yy}") if int(yy) < 50 else int(f"19{yy}")  # 2025 or 2005
            month_num = f"{int(mm):02d}"  # "01", "02", "12", etc.
            full_date = f"{year}{mm}{dd}"  # "20251231" or "20051231"
            
            # Convert symbol to lowercase
            symbol_lower = symbol.lower()
            
            # Determine type folder and filename for raw_data structure
            if instrument_type == "CE":
                # raw_data/option/{symbol}_call/{year}/{month}/{symbol}_call_{YYYYMMDD}.parquet
                folder_path = os.path.join(output_folder, "option", f"{symbol_lower}_call", str(year), month_num)
                filename = f"{symbol_lower}_call_{full_date}.parquet"
            elif instrument_type == "PE":
                # raw_data/option/{symbol}_put/{year}/{month}/{symbol}_put_{YYYYMMDD}.parquet
                folder_path = os.path.join(output_folder, "option", f"{symbol_lower}_put", str(year), month_num)
                filename = f"{symbol_lower}_put_{full_date}.parquet"
            elif instrument_type == "CASH":
                # raw_data/cash/{symbol}/{year}/{month}/{symbol}_cash_{YYYYMMDD}.parquet
                folder_path = os.path.join(output_folder, "cash", symbol_lower, str(year), month_num)
                filename = f"{symbol_lower}_cash_{full_date}.parquet"
            else:  # FUTURE
                # raw_data/futures/{symbol}/{year}/{month}/{symbol}_future_{YYYYMMDD}.parquet
                folder_path = os.path.join(output_folder, "futures", symbol_lower, str(year), month_num)
                filename = f"{symbol_lower}_future_{full_date}.parquet"
            
            os.makedirs(folder_path, exist_ok=True)
            out_file = os.path.join(folder_path, filename)
            
            # Save as parquet using pyarrow
            table = pa.Table.from_pandas(df_date, preserve_index=False)
            pq.write_table(table, out_file, compression="snappy")
            logger.info(f"Saved transformed data: {out_file} ({len(df_date)} records)")

    def _process_futures_transformed(self, symbol, future_dataframes, output_folder):
        """Process futures data - keep nearest expiry per trade date, then merge"""
        if not future_dataframes:
            return
        
        # Group futures by trade date (based on dates in the data)
        # Structure: {trade_date_str: [(df, expiry_date), ...]}
        futures_by_trade_date = {}
        
        for fut_data, expiry_date in future_dataframes:
            if fut_data.empty:
                continue
            
            # Get trade dates from the dataframe
            df_work = fut_data.copy()
            if isinstance(df_work['date'].dtype, pd.DatetimeTZDtype):
                df_work['date'] = df_work['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            
            df_work['trade_date'] = df_work['date'].dt.date
            
            for trade_date, day_df in df_work.groupby('trade_date'):
                date_str = trade_date.strftime('%Y%m%d')
                
                if date_str not in futures_by_trade_date:
                    futures_by_trade_date[date_str] = []
                
                futures_by_trade_date[date_str].append((day_df, expiry_date))
        
        # For each trade date, keep only the nearest expiry, then merge all
        all_processed_dfs = []
        
        for date_str, date_futures in futures_by_trade_date.items():
            if not date_futures:
                continue
            
            # Sort by expiry date and keep the nearest one (first after sorting)
            date_futures.sort(key=lambda x: x[1])
            nearest_expiry_data = date_futures[0]  # (day_df, expiry_date)
            all_processed_dfs.append(nearest_expiry_data)
        
        # Now merge all processed futures and transform
        if all_processed_dfs:
            df_list = []
            for fut_df, expiry_date in all_processed_dfs:
                transformed_df = self._transform_dataframe(fut_df, symbol, "FUT", None, expiry_date)
                if not transformed_df.empty:
                    df_list.append(transformed_df)
            
            if df_list:
                merged_df = pd.concat(df_list, ignore_index=True)
                merged_df = self._calculate_coi(merged_df, "FUT")
                
                # Split by date and save
                for date, df_date in merged_df.groupby("date"):
                    merge_date = str(date)  # YYMMDD format
                    
                    # Extract year, month, and full date from YYMMDD
                    # Zero-pad to ensure 6 digits
                    date_str = f"{int(date):06d}"  # "251231" or "051231"
                    yy = date_str[:2]     # "25" or "05"
                    mm = date_str[2:4]    # "12" or "12"
                    dd = date_str[4:6]    # "31" or "31"
                    year = int(f"20{yy}") if int(yy) < 50 else int(f"19{yy}")  # 2025 or 2005
                    month_num = f"{int(mm):02d}"  # "01", "02", "12", etc.
                    full_date = f"{year}{mm}{dd}"  # "20251231" or "20051231"
                    
                    # Convert symbol to lowercase
                    symbol_lower = symbol.lower()
                    
                    # Build folder path: raw_data/futures/{symbol}/{year}/{month}/
                    folder_path = os.path.join(output_folder, "futures", symbol_lower, str(year), month_num)
                    filename = f"{symbol_lower}_future_{full_date}.parquet"
                    os.makedirs(folder_path, exist_ok=True)
                    out_file = os.path.join(folder_path, filename)
                    
                    table = pa.Table.from_pandas(df_date, preserve_index=False)
                    pq.write_table(table, out_file, compression="snappy")
                    logger.info(f"Saved transformed FUTURES data: {out_file} ({len(df_date)} records)")

    def _process_options_transformed(self, symbol, options_dataframes, option_type, output_folder):
        """Process options data (CE or PE) - merge all strikes/expiries"""
        if not options_dataframes:
            return
        
        df_list = []
        
        for opt_data, strike, expiry_date in options_dataframes:
            if opt_data.empty:
                continue
            
            # Apply transformations
            transformed_df = self._transform_dataframe(opt_data, symbol, option_type, strike, expiry_date)
            if not transformed_df.empty:
                df_list.append(transformed_df)
        
        if not df_list:
            return
        
        # Merge all dataframes
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # Calculate COI
        merged_df = self._calculate_coi(merged_df, option_type)
        
        # Split by date and save
        for date, df_date in merged_df.groupby("date"):
            merge_date = str(date)  # YYMMDD format
            
            # Extract year, month, and full date from YYMMDD
            # Zero-pad to ensure 6 digits
            date_str = f"{int(date):06d}"  # "251231" or "051231"
            yy = date_str[:2]     # "25" or "05"
            mm = date_str[2:4]    # "12" or "12"
            dd = date_str[4:6]    # "31" or "31"
            year = int(f"20{yy}") if int(yy) < 50 else int(f"19{yy}")  # 2025 or 2005
            month_num = f"{int(mm):02d}"  # "01", "02", "12", etc.
            full_date = f"{year}{mm}{dd}"  # "20251231" or "20051231"
            
            # Convert symbol to lowercase
            symbol_lower = symbol.lower()
            
            # Determine type folder and filename for raw_data structure
            if option_type == "CE":
                # raw_data/option/{symbol}_call/{year}/{month}/{symbol}_call_{YYYYMMDD}.parquet
                folder_path = os.path.join(output_folder, "option", f"{symbol_lower}_call", str(year), month_num)
                filename = f"{symbol_lower}_call_{full_date}.parquet"
            else:  # PE
                # raw_data/option/{symbol}_put/{year}/{month}/{symbol}_put_{YYYYMMDD}.parquet
                folder_path = os.path.join(output_folder, "option", f"{symbol_lower}_put", str(year), month_num)
                filename = f"{symbol_lower}_put_{full_date}.parquet"
            
            os.makedirs(folder_path, exist_ok=True)
            out_file = os.path.join(folder_path, filename)
            
            table = pa.Table.from_pandas(df_date, preserve_index=False)
            pq.write_table(table, out_file, compression="snappy")
            logger.info(f"Saved transformed {option_type} data: {out_file} ({len(df_date)} records)")

    def process_saved_files_for_transformation(self, input_folder, output_folder, from_date=None, to_date=None):
        """Process all saved parquet files and create transformed format
        
        Args:
            input_folder: Input folder path (raw_parquet)
            output_folder: Output folder path (raw_data)
            from_date: Start date for filtering (only process files from this date)
            to_date: End date for filtering (only process files up to this date)
        """
        from glob import glob
        
        logger.info("Starting transformation of saved files...")
        if from_date and to_date:
            logger.info(f"Filtering files by date range: {from_date} to {to_date}")
        
        # Check if input folder exists
        if not os.path.exists(input_folder):
            logger.error(f"Input folder does not exist: {input_folder}")
            return
        
        # Process each instrument type
        for option_type in ["CE", "PE", "FUTURE", "CASH"]:
            logger.info(f"\nProcessing {option_type} files...")
            
            # Read all parquet files for this type (with date filtering)
            all_files = self._read_parquet_files_from_date_folders(input_folder, option_type, from_date, to_date)
            logger.info(f"Found {len(all_files)} Parquet file(s) for {option_type}")
            
            if not all_files:
                continue
            
            # Group files by stock
            stock_files = {}
            for file in all_files:
                stock, _, _ = self._extract_info_from_filename(file, option_type)
                if not stock:
                    continue
                stock_files.setdefault(stock, []).append(file)
            
            # Process each stock
            for stock, files in stock_files.items():
                # FUTURE: Group by date folder first, then keep nearest expiry per date folder
                if option_type == "FUTURE":
                    files_by_date_folder = {}
                    for f in files:
                        # Extract date folder from path: raw_parquet/YEAR/MONTH/YYYYMMDD/TYPE/
                        # Get the YYYYMMDD folder name
                        path_parts = Path(f).parts
                        # Find the 8-digit date folder in the path
                        date_folder = None
                        for part in path_parts:
                            if part.isdigit() and len(part) == 8:
                                date_folder = part
                                break
                        if date_folder and date_folder not in files_by_date_folder:
                            files_by_date_folder[date_folder] = []
                        if date_folder:
                            files_by_date_folder[date_folder].append(f)
                    
                    processed_files = []
                    for date_folder, date_files in files_by_date_folder.items():
                        file_expiries = []
                        for f in date_files:
                            _, _, exp = self._extract_info_from_filename(f, option_type)
                            if exp:
                                try:
                                    exp_dt = pd.to_datetime(exp)
                                    file_expiries.append((f, exp_dt))
                                except:
                                    file_expiries.append((f, exp))
                        if file_expiries:
                            file_expiries.sort(key=lambda x: x[1])
                            processed_files.append(file_expiries[0][0])
                        else:
                            processed_files.extend(date_files)
                    
                    files = processed_files
                    logger.info(f"Processing stock: {stock} → {len(files)} file(s) from {len(files_by_date_folder)} date folder(s)")
                else:
                    logger.info(f"Processing stock: {stock} → {len(files)} file(s)")
                
                df_list = []
                
                for file in files:
                    try:
                        df = pd.read_parquet(file)
                        if df.empty:
                            continue
                        
                        # Extract info from filename
                        stock_name, strike, expiry = self._extract_info_from_filename(file, option_type)
                        
                        # Apply transformations
                        transformed_df = self._transform_dataframe(df, stock_name, option_type, strike, expiry)
                        if not transformed_df.empty:
                            df_list.append(transformed_df)
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {e}")
                        continue
                
                if not df_list:
                    logger.warning(f"No valid data for stock: {stock}")
                    continue
                
                # Merge all dataframes
                merged_df = pd.concat(df_list, ignore_index=True)
                
                # Calculate COI
                merged_df = self._calculate_coi(merged_df, option_type)
                
                # Split by date and save
                for date, df_date in merged_df.groupby("date"):
                    merge_date = str(date)  # YYMMDD format
                    
                    # Extract year, month, and full date from YYMMDD
                    # Zero-pad to ensure 6 digits
                    date_str = f"{int(date):06d}"  # "251231" or "051231"
                    yy = date_str[:2]     # "25" or "05"
                    mm = date_str[2:4]    # "12" or "12"
                    dd = date_str[4:6]    # "31" or "31"
                    year = int(f"20{yy}") if int(yy) < 50 else int(f"19{yy}")  # 2025 or 2005
                    month_num = f"{int(mm):02d}"  # "01", "02", "12", etc.
                    full_date = f"{year}{mm}{dd}"  # "20251231" or "20051231"
                    
                    # Convert stock to lowercase
                    stock_lower = stock.lower()
                    
                    # Determine type folder and filename for raw_data structure
                    if option_type == "CE":
                        # raw_data/option/{symbol}_call/{year}/{month}/{symbol}_call_{YYYYMMDD}.parquet
                        folder_path = os.path.join(output_folder, "option", f"{stock_lower}_call", str(year), month_num)
                        filename = f"{stock_lower}_call_{full_date}.parquet"
                    elif option_type == "PE":
                        # raw_data/option/{symbol}_put/{year}/{month}/{symbol}_put_{YYYYMMDD}.parquet
                        folder_path = os.path.join(output_folder, "option", f"{stock_lower}_put", str(year), month_num)
                        filename = f"{stock_lower}_put_{full_date}.parquet"
                    elif option_type == "CASH":
                        # raw_data/cash/{symbol}/{year}/{month}/{symbol}_cash_{YYYYMMDD}.parquet
                        folder_path = os.path.join(output_folder, "cash", stock_lower, str(year), month_num)
                        filename = f"{stock_lower}_cash_{full_date}.parquet"
                    else:  # FUTURE
                        # raw_data/futures/{symbol}/{year}/{month}/{symbol}_future_{YYYYMMDD}.parquet
                        folder_path = os.path.join(output_folder, "futures", stock_lower, str(year), month_num)
                        filename = f"{stock_lower}_future_{full_date}.parquet"
                    
                    os.makedirs(folder_path, exist_ok=True)
                    out_file = os.path.join(folder_path, filename)
                    
                    table = pa.Table.from_pandas(df_date, preserve_index=False)
                    pq.write_table(table, out_file, compression="snappy")
                    logger.info(f"Saved transformed data: {out_file} ({len(df_date)} records)")
        
        logger.info("Transformation of saved files completed.")

def copy_transformed_data_to_local(transformed_folder: str, output_dir: str = "/home/ubuntu/raw_data", from_date=None, to_date=None):
    """
    Copy transformed parquet files to local directory using SSH upload structure.
    Only copies files within the specified date range (defaults to today if not specified).
    
    Folder structure:
    - cash/{symbol}/{year}/{month}/{symbol}_cash_{YYYYMMDD}.parquet
    - futures/{symbol}/{year}/{month}/{symbol}_future_{YYYYMMDD}.parquet
    - option/{symbol}_call/{year}/{month}/{symbol}_call_{YYYYMMDD}.parquet
    - option/{symbol}_put/{year}/{month}/{symbol}_put_{YYYYMMDD}.parquet
    
    Args:
        transformed_folder: Path to the transformed finaloutput folder
        output_dir: Output directory path (default: /home/ubuntu/raw_data)
        from_date: Start date (datetime object or None for today)
        to_date: End date (datetime object or None for today)
    """
    local_root = Path(transformed_folder)
    if not local_root.exists():
        logger.error("Transformed folder does not exist: %s", transformed_folder)
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set date range - default to today if not specified
    if from_date is None:
        today = datetime.now().date()
        from_date = datetime.combine(today, dt_time(0, 0, 0))
    if to_date is None:
        today = datetime.now().date()
        to_date = datetime.combine(today, dt_time(23, 59, 59))
    
    # Convert to date objects for comparison
    if isinstance(from_date, datetime):
        from_date_date = from_date.date()
    else:
        from_date_date = from_date
    
    if isinstance(to_date, datetime):
        to_date_date = to_date.date()
    else:
        to_date_date = to_date
    
    logger.info("Copying data for date range: %s to %s", from_date_date, to_date_date)
    
    files_copied = 0
    files_by_type = {}
    
    # Map folder names to data types
    folder_to_data_type = {
        "CALL": "option/call",
        "PUT": "option/put",
        "CASH": "cash",
        "FUTURES": "futures",
    }
    
    # Traverse YEAR folders
    for year_item in os.listdir(local_root):
        year_path = local_root / year_item
        if not year_path.is_dir() or not year_item.isdigit() or len(year_item) != 4:
            continue
        
        year = year_item
        
        # Traverse MONTH folders
        for month_item in os.listdir(year_path):
            month_path = year_path / month_item
            if not month_path.is_dir() or not month_item.isdigit() or len(month_item) != 2:
                continue
            
            month = month_item
            
            # Traverse YYYYMMDD date folders
            for date_item in os.listdir(month_path):
                date_path = month_path / date_item
                if not date_path.is_dir() or not date_item.isdigit() or len(date_item) != 8:
                    continue
                
                # Extract date from folder name (YYYYMMDD)
                try:
                    folder_date = datetime.strptime(date_item, "%Y%m%d").date()
                    dd = date_item[6:8]  # Day
                except ValueError:
                    logger.warning("Invalid date folder name: %s", date_item)
                    continue
                
                # Skip if folder date is outside the specified date range
                if folder_date < from_date_date or folder_date > to_date_date:
                    logger.debug("Skipping folder %s (date %s outside range %s to %s)", 
                                date_item, folder_date, from_date_date, to_date_date)
                    continue
                
                # Traverse TYPE folders (CASH, FUTURES, OPTION/CALL, OPTION/PUT)
                for type_folder_name in ["CASH", "FUTURES", "OPTION/CALL", "OPTION/PUT"]:
                    type_path = date_path / type_folder_name
                    if not type_path.exists() or not type_path.is_dir():
                        continue
                    
                    # Determine data type
                    if type_folder_name == "OPTION/CALL":
                        data_type = "option/call"
                        folder_name = "CALL"
                    elif type_folder_name == "OPTION/PUT":
                        data_type = "option/put"
                        folder_name = "PUT"
                    elif type_folder_name == "CASH":
                        data_type = "cash"
                        folder_name = "CASH"
                    elif type_folder_name == "FUTURES":
                        data_type = "futures"
                        folder_name = "FUTURES"
                    else:
                        continue
                    
                    # Find all parquet files in this type folder
                    parquet_files = sorted(type_path.glob("*.parquet"))
                    
                    for parquet_path in parquet_files:
                        filename = parquet_path.stem
                        
                        # Extract symbol and date from filename
                        # Patterns:
                        # - SYMBOL_CALL_YYMMDD.parquet
                        # - SYMBOL_PUT_YYMMDD.parquet
                        # - SYMBOL_YYMMDD.parquet (for CASH and FUTURES)
                        
                        if folder_name in ["CALL", "PUT"]:
                            # Pattern: SYMBOL_CALL_YYMMDD or SYMBOL_PUT_YYMMDD
                            suffix_pattern = f"_{folder_name}_"
                            if suffix_pattern in filename:
                                parts = filename.split(suffix_pattern, 1)
                                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
                                    base_name = parts[0]
                                else:
                                    logger.warning("Could not parse filename (expected SYMBOL_%s_YYMMDD): %s", folder_name, filename)
                                    continue
                            else:
                                # Try alternative: split by last underscore before 6-digit date
                                parts = filename.rsplit("_", 1)
                                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
                                    symbol_part = parts[0]
                                    suffix = f"_{folder_name}"
                                    if symbol_part.endswith(suffix):
                                        base_name = symbol_part[:-len(suffix)]
                                    else:
                                        base_name = symbol_part
                                else:
                                    logger.warning("Could not parse filename (expected SYMBOL_%s_YYMMDD): %s", folder_name, filename)
                                    continue
                        else:
                            # For CASH and FUTURES: SYMBOL_YYMMDD
                            parts = filename.rsplit("_", 1)
                            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
                                base_name = parts[0]
                            else:
                                logger.warning("Could not parse filename: %s", filename)
                                continue
                        
                        symbol = base_name.lower()
                        
                        # Extract date from filename (6-digit pattern YYMMDD)
                        date_match_6 = re.search(r'(\d{6})$', filename)
                        if not date_match_6:
                            logger.warning("Could not extract date from filename: %s", filename)
                            continue
                        
                        date_6 = date_match_6.group(1)
                        file_yy = date_6[:2]
                        file_mm = date_6[2:4]
                        file_dd = date_6[4:6]
                        
                        # Validate it looks like a date
                        if not (1 <= int(file_mm) <= 12 and 1 <= int(file_dd) <= 31):
                            logger.warning("Invalid date in filename: %s", filename)
                            continue
                        
                        # Construct YYYYMMDD from year folder and date_6
                        yyyymmdd = f"{year}{file_mm}{file_dd}"
                        
                        # Generate target filename
                        if data_type == "cash":
                            target_filename = f"{symbol}_cash_{yyyymmdd}.parquet"
                        elif data_type == "futures":
                            target_filename = f"{symbol}_future_{yyyymmdd}.parquet"
                        elif data_type == "option/call":
                            target_filename = f"{symbol}_call_{yyyymmdd}.parquet"
                        elif data_type == "option/put":
                            target_filename = f"{symbol}_put_{yyyymmdd}.parquet"
                        else:
                            continue
                        
                        # Build target directory path
                        if data_type.startswith("option/"):
                            option_type = data_type.split("/")[1]  # "call" or "put"
                            target_dir = output_path / "option" / f"{symbol}_{option_type}" / year / month
                        else:
                            target_dir = output_path / data_type / symbol / year / month
                        
                        # Create target directory
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        target_path = target_dir / target_filename
                        shutil.copy2(parquet_path, target_path)
                        files_copied += 1
                        files_by_type[data_type] = files_by_type.get(data_type, 0) + 1
                        
                        if files_copied % 10 == 0:
                            logger.debug("Copied %d files...", files_copied)
    
    logger.info("Copied %d parquet files to %s", files_copied, output_dir)
    logger.info("Breakdown by data type:")
    for data_type, count in files_by_type.items():
        logger.info("  - %s: %d files", data_type, count)
    
    return True

def filter_symbols_with_lot_size(symbols: list, csv_path: Path) -> list:
    """
    Filter symbols to only include those that have lot sizes in the CSV.
    Symbols without lot sizes will be skipped (not processed with default lot size 1).
    
    Args:
        symbols: List of stock symbols to filter
        csv_path: Path to the lot size CSV file
    
    Returns:
        List of symbols that have lot sizes in the CSV (empty list if CSV not found)
    """
    if not csv_path.exists():
        logger.error(f"Lot size CSV not found at {csv_path}. Cannot filter symbols. Returning empty list.")
        return []
    
    try:
        import pandas as pd
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Find the stock name column
        stock_column = None
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['symbol', 'stock', 'name', 'instrument']):
                stock_column = col
                break
        
        if stock_column is None:
            stock_column = df.columns[0]
        
        # Find lot size column - try "Nov 2025" or "November 2025"
        lot_size_column = None
        for col in df.columns:
            col_lower = col.lower()
            if ('nov' in col_lower and '2025' in col_lower) or ('lot' in col_lower and 'size' in col_lower):
                lot_size_column = col
                break
        
        if lot_size_column is None:
            logger.error(f"Lot size column not found in CSV. Available columns: {list(df.columns)}")
            logger.error("Cannot filter symbols without lot size column. Returning empty list.")
            return []
        
        # Create set of symbols that have valid lot sizes
        valid_symbols = set()
        for _, row in df.iterrows():
            symbol = str(row[stock_column]).strip().upper()
            lot_size = row[lot_size_column]
            
            # Check if lot size is valid (not NaN, not None, > 0)
            if pd.notna(lot_size) and lot_size is not None:
                try:
                    lot_size_int = int(float(lot_size))
                    if lot_size_int > 0:
                        valid_symbols.add(symbol)
                except (ValueError, TypeError):
                    continue
        
        # Filter input symbols to only include those with valid lot sizes
        filtered_symbols = [s.upper() for s in symbols if s.upper() in valid_symbols]
        
        skipped_count = len(symbols) - len(filtered_symbols)
        if skipped_count > 0:
            logger.info(f"Filtered {len(filtered_symbols)} symbols with lot sizes from {len(symbols)} total symbols. Skipped {skipped_count} symbols without lot sizes.")
        else:
            logger.info(f"All {len(filtered_symbols)} symbols have valid lot sizes in CSV.")
        
        return filtered_symbols
        
    except Exception as e:
        logger.error(f"Error filtering symbols by lot size: {e}")
        logger.error(traceback.format_exc())
        # On error, return empty list to avoid processing symbols without lot sizes
        logger.error("Returning empty list due to filtering error - will skip all symbols")
        return []


def run_nautilus_transformation(symbols: list, current_date: date) -> bool:
    """
    Run Nautilus transformation on current day's data.
    
    Args:
        symbols: List of stock symbols to transform
        current_date: Current date to process (YYYY-MM-DD)
    
    Returns:
        True if transformation succeeded, False otherwise
    """
    # Prepare date string for use in error messages
    date_str = current_date.strftime("%Y-%m-%d")
    
    try:
        # Construct path to transform_nautilus.py script
        transform_script = Path(script_dir) / "marvelquant" / "scripts" / "transformation" / "transformers" / "transform_nautilus.py"
        
        if not transform_script.exists():
            logger.error("Nautilus transformation script not found at: %s", transform_script)
            send_telegram_alert(
                f"❌ Zerodha stocks data: Nautilus transformation script not found at {transform_script}"
            )
            return False
        
        # Prepare arguments
        input_dir = "/home/ubuntu/raw_data"
        output_dir = "/data/nautilus_data"
        
        # Path to lot size CSV - resolve to absolute path
        lot_size_csv = Path("/home/ubuntu/Stock_automation/marvelquant/NseLotSize.csv").resolve()
        
        # Verify CSV file exists
        if not lot_size_csv.exists():
            logger.error("Lot size CSV file not found at: %s", lot_size_csv)
            send_telegram_alert(
                f"❌ Zerodha stocks data: Lot size CSV not found at {lot_size_csv}\n"
                f"📅 Date: {date_str}\n"
                f"⚠️ Skipping Nautilus transformation"
            )
            return False
        
        logger.info("Using lot size CSV: %s", lot_size_csv)
        
        # Filter symbols to only include those with lot sizes in CSV
        logger.info("Filtering symbols to only include those with lot sizes in CSV...")
        filtered_symbols = filter_symbols_with_lot_size(symbols, lot_size_csv)
        
        if not filtered_symbols:
            logger.error("No symbols with valid lot sizes found. Skipping Nautilus transformation.")
            send_telegram_alert(
                f"❌ Zerodha stocks data: No symbols with valid lot sizes found in CSV.\n"
                f"📅 Date: {date_str}\n"
                f"⚠️ Skipping Nautilus transformation"
            )
            return False
        
        # Convert symbols to uppercase (already done in filter, but ensure)
        symbol_list = [s.upper() for s in filtered_symbols]
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build command - use absolute resolved path for CSV
        python_path = "/usr/bin/python3"
        # Ensure types are clean strings (no non-breaking spaces)
        types_list = ["options", "equity", "futures"]
        # Clean any potential non-breaking spaces
        types_list = [t.replace('\xa0', ' ').replace('\u00a0', ' ').strip() for t in types_list]
        
        cmd = [
            python_path,
            str(transform_script),
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--symbols"] + symbol_list + [
            "--start-date", date_str,
            "--end-date", date_str,
            "--types"] + types_list + [
            "--lot-size-csv", str(lot_size_csv),
            "--log-level", "INFO"
        ]
        
        logger.info("Starting Nautilus transformation...")
        logger.info("  Input directory: %s", input_dir)
        logger.info("  Output directory: %s", output_dir)
        logger.info("  Date: %s", date_str)
        logger.info("  Symbols: %s", ", ".join(symbol_list[:10]) + ("..." if len(symbol_list) > 10 else ""))
        logger.info("  Total symbols: %d (filtered from %d original symbols)", len(symbol_list), len(symbols))
        logger.info("  Lot size CSV: %s", lot_size_csv)
        
        # Debug: Log the command to verify it's correct
        logger.debug("Command being executed: %s", " ".join(cmd))
        # Check for any non-breaking spaces in command
        for i, arg in enumerate(cmd):
            if '\xa0' in arg or '\u00a0' in arg:
                logger.error("⚠️  Non-breaking space found in command argument %d: %s", i, repr(arg))
        
        send_telegram_alert(
            f"🔄 Zerodha stocks data: Starting Nautilus transformation...\n"
            f"📅 Date: {date_str}\n"
            f"📊 Symbols: {len(symbol_list)} symbols (filtered from {len(symbols)} original)\n"
            f"📁 Output: {output_dir}"
        )
        
        # Run transformation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(script_dir),
            timeout=28800  # 8 hours timeout (for processing all 196 symbols)
        )
        
        if result.returncode == 0:
            logger.info("Nautilus transformation completed successfully.")
            logger.debug("Transformation output:\n%s", result.stdout)
            send_telegram_alert(
                f"✅ Zerodha stocks data: Nautilus transformation completed successfully!\n"
                f"📅 Date: {date_str}\n"
                f"📊 Processed {len(symbol_list)} symbols\n"
                f"📁 Output: {output_dir}"
            )
            return True
        else:
            logger.error("Nautilus transformation failed with return code: %d", result.returncode)
            logger.error("Error output:\n%s", result.stderr)
            logger.error("Standard output:\n%s", result.stdout)
            send_telegram_alert(
                f"❌ Zerodha stocks data: Nautilus transformation failed!\n"
                f"📅 Date: {date_str}\n"
                f"⚠️ Return code: {result.returncode}\n"
                f"💡 Check logs for details"
            )
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Nautilus transformation timed out after 8 hours")
        send_telegram_alert(
            f"❌ Zerodha stocks data: Nautilus transformation timed out!\n"
            f"📅 Date: {date_str}\n"
            f"⏱️ Timeout: 8 hours"
        )
        return False
    except Exception as e:
        logger.error("Error running Nautilus transformation: %s", str(e))
        logger.error(traceback.format_exc())
        send_telegram_alert(
            f"❌ Zerodha stocks data: Error in Nautilus transformation!\n"
            f"📅 Date: {date_str}\n"
            f"⚠️ Error: {str(e)}"
        )
        return False

def run_nautilus_only(date_str: str = None) -> bool:
    """
    Run only the Nautilus transformation for manual runs.
    
    Args:
        date_str: Date to process in YYYY-MM-DD format (defaults to today)
    
    Returns:
        True if transformation succeeded, False otherwise
    """
    try:
        # Set up logging
        global today
        if date_str:
            today = date_str
        else:
            today = datetime.now().strftime("%Y-%m-%d")
        logfile(filename=os.path.join(script_dir, f'logfile_nautilus_only_{today}.log'))
        
        logger.info("="*50)
        logger.info("Running Nautilus transformation only (manual run)")
        logger.info("="*50)
        
        # Parse date
        if date_str:
            current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            current_date = datetime.now().date()
        
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Processing date: {date_str}")
        
        # Read stock symbols from CSV
        stock_file = os.path.join(script_dir, "stock_namess.csv")
        if not os.path.exists(stock_file):
            logger.error(f"stock_namess.csv not found at {stock_file}")
            send_telegram_alert(
                f"❌ Zerodha stocks data: 'stock_namess.csv' not found at {stock_file}\n"
                f"📅 Date: {date_str}\n"
                f"⚠️ Cannot run Nautilus transformation"
            )
            return False
        
        stock_df = pd.read_csv(stock_file)
        stock_symbols = stock_df['stock_name'].dropna().unique()
        total_stocks = len(stock_symbols)
        
        logger.info(f"Found {total_stocks} stocks in CSV file")
        
        # Send initial alert
        send_telegram_alert(
            f"🔄 Zerodha stocks data: Starting Nautilus transformation (manual run)...\n"
            f"📅 Date: {date_str}\n"
            f"📊 Processing {total_stocks} symbols\n"
            f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Run Nautilus transformation
        nautilus_success = run_nautilus_transformation(stock_symbols, current_date)
        
        if nautilus_success:
            logger.info("Nautilus transformation completed successfully.")
            send_telegram_alert(
                f"✅ Zerodha stocks data: Nautilus transformation completed successfully!\n"
                f"📅 Date: {date_str}\n"
                f"📊 Processed {total_stocks} symbols\n"
                f"📁 Output: /data/nautilus_data/data"
            )
            return True
        else:
            logger.error("Nautilus transformation failed.")
            return False
            
    except Exception as e:
        logger.error("Error in Nautilus-only run: %s", str(e))
        logger.error(traceback.format_exc())
        send_telegram_alert(
            f"❌ Zerodha stocks data: Error in Nautilus-only run!\n"
            f"📅 Date: {date_str if 'date_str' in locals() else 'N/A'}\n"
            f"⚠️ Error: {str(e)}"
        )
        return False


def is_trading_day(date_to_check=None):
    """Check if given date is a trading day (not weekend, not holiday)"""
    if date_to_check is None:
        date_to_check = datetime.now().date()
    elif isinstance(date_to_check, datetime):
        date_to_check = date_to_check.date()
    
    # Check if weekend (Saturday=5, Sunday=6)
    if date_to_check.weekday() >= 5:
        logger.info(f"{date_to_check} is a weekend. Not a trading day.")
        return False
    
    # Check holidays CSV
    holiday_file = os.path.join(script_dir, "nse_holidays.csv")
    if os.path.exists(holiday_file):
        try:
            holidays_df = pd.read_csv(holiday_file)
            
            # Parse dates from DD-MM-YYYY format
            holiday_dates = []
            for date_str in holidays_df['date'].dropna():
                try:
                    # Try DD-MM-YYYY format
                    parsed_date = datetime.strptime(str(date_str).strip(), "%d-%m-%Y").date()
                    holiday_dates.append(parsed_date)
                except ValueError:
                    try:
                        # Try YYYY-MM-DD format as fallback
                        parsed_date = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").date()
                        holiday_dates.append(parsed_date)
                    except ValueError:
                        logger.warning(f"Could not parse holiday date: {date_str}")
                        continue
            
            # Check if date is in holidays list
            if date_to_check in holiday_dates:
                logger.info(f"{date_to_check} is a holiday. Not a trading day.")
                return False
        except Exception as e:
            logger.warning(f"Error reading holiday file {holiday_file}: {e}. Continuing without holiday check.")
    else:
        logger.warning(f"Holiday file not found at {holiday_file}. Continuing without holiday check.")
    
    return True

def get_access_token_only():
    """Function to just get and update access token without fetching data"""
    try:
        logger.info("Getting access token only (no data fetching)...")
        
        login_result = auto_login(
            CREDENTIALS["user_id"], 
            CREDENTIALS["password"], 
            CREDENTIALS["totp_key"], 
            CREDENTIALS["api_key"], 
            CREDENTIALS["api_secret"]
        )
        
        if login_result is None:
            logger.error("Login failed. Cannot update access token.")
            return False
        
        request_token, access_token, kite = login_result
        logger.info("Login successful. Access token obtained.")
        
        update_env_file(CREDENTIALS["api_key"], access_token)
        logger.info("Access token updated in .env file. fetch_historical_data_cu.py can now use it.")
        return True
        
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        logger.error(traceback.format_exc())
        return False

def run_data_fetching_session(from_date=None, to_date=None):
    """Main function to run data fetching session"""
    global today
    today = datetime.now().strftime("%Y-%m-%d")
    logfile(filename=os.path.join(script_dir, f'logfile_auto_login_{today}.log'))
    
    try:
        logger.info("Starting data fetching session...")

        # Login and get access token
        login_result = auto_login(
            CREDENTIALS["user_id"], 
            CREDENTIALS["password"], 
            CREDENTIALS["totp_key"], 
            CREDENTIALS["api_key"], 
            CREDENTIALS["api_secret"]
        )
        
        if login_result is None:
            logger.error("Login failed. Cannot proceed with data fetching.")
            send_telegram_alert("❌ Zerodha stocks data: login failed, cannot proceed with data fetching.")
            return
        
        request_token, access_token, kite = login_result
        logger.info("Login successful. Access token obtained.")
        
        # Update .env file with access token
        update_env_file(CREDENTIALS["api_key"], access_token)
        logger.info("Access token updated in .env file.")
        send_telegram_alert(f"✅ Zerodha stocks data: access token generated and .env updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        
        # Set access token for kite instance
        kite.set_access_token(access_token)
        
        # Initialize data fetcher
        fetcher = ZerodhaDataFetcher(kite)
        
        # Set date range - default to current day only
        if from_date is None:
            today = datetime.now().date()
            from_date = datetime.combine(today, dt_time(0, 0, 0))  # Today 00:00:00
        if to_date is None:
            today = datetime.now().date()
            to_date = datetime.combine(today, dt_time(23, 59, 59))  # Today 23:59:59
        
        # Read stock symbols
        stock_file = os.path.join(script_dir, "stock_namess.csv")
        if not os.path.exists(stock_file):
            logger.error(f"stock_namess.csv not found at {stock_file}")
            send_telegram_alert("❌ Zerodha stocks data: 'stock_namess.csv' not found, aborting.")
            return
        
        stock_df = pd.read_csv(stock_file)
        stock_symbols = stock_df['stock_name'].dropna().unique()
        total_stocks = len(stock_symbols)
        
        logger.info(f"Found {total_stocks} stocks to process")
        send_telegram_alert(
            f"ℹ️ Zerodha stocks data: starting download for {total_stocks} stocks "
            f"from {from_date} to {to_date}."
        )
        
        # Fetch data for each symbol
        for idx, symbol in enumerate(stock_symbols, start=1):
            symbol = symbol.upper()
            logger.info(f"\nProcessing {symbol} ({idx}/{total_stocks})")
            fetcher.fetch_historical_data(symbol, from_date, to_date)
            time.sleep(0.5)
        
        logger.info("Data fetching session completed successfully.")
        
        # Now process all saved files for transformation
        logger.info("\n" + "="*50)
        logger.info("Starting transformation of downloaded files...")
        logger.info("="*50)
        
        send_telegram_alert(
            f"🔄 Zerodha stocks data: Download completed for {total_stocks} stocks. "
            f"Starting transformation of downloaded files..."
        )
        
        output_base = "raw_parquet"
        input_folder = output_base
        output_folder = "/home/ubuntu/raw_data"
        
        # Pass date range to transformation to only process today's data
        # Data will be saved directly to raw_data folder with correct structure
        fetcher.process_saved_files_for_transformation(input_folder, output_folder, from_date, to_date)
        
        logger.info("Transformation completed successfully.")
        
        # Format date range for display
        from_date_str = from_date.strftime("%Y-%m-%d") if isinstance(from_date, datetime) else str(from_date)
        to_date_str = to_date.strftime("%Y-%m-%d") if isinstance(to_date, datetime) else str(to_date)
        
        send_telegram_alert(
            f"✅ Zerodha stocks data: Transformation completed successfully.\n"
            f"📁 Data saved directly to: /home/ubuntu/raw_data\n"
            f"📊 {total_stocks} stocks processed for date range {from_date_str} to {to_date_str}.\n"
            f"⏰ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Extract current date from from_date (should be today for single day runs)
        if isinstance(from_date, datetime):
            current_date = from_date.date()
        elif isinstance(from_date, str):
            current_date = datetime.strptime(from_date, "%Y-%m-%d").date()
        else:
            current_date = datetime.now().date()
        
        # Run Nautilus transformation
        # Nautilus transformation creates a different output format for NautilusTrader
        # It reads from the same SSH server location but outputs to a different location
        logger.info("\n" + "="*50)
        logger.info("Starting Nautilus transformation...")
        logger.info("="*50)
        
        # Send alert before starting Nautilus transformation
        send_telegram_alert(
            f"🔄 Zerodha stocks data: Starting Nautilus transformation...\n"
            f"📅 Date: {current_date.strftime('%Y-%m-%d')}\n"
            f"📊 Processing {len(stock_symbols)} symbols\n"
            f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        nautilus_success = run_nautilus_transformation(stock_symbols, current_date)
        
        if nautilus_success:
            logger.info("Nautilus transformation completed successfully.")
        else:
            logger.warning("Nautilus transformation failed, but workflow continues.")
        
    except Exception as e:
        logger.error("Error in data fetching session:")
        logger.error(traceback.format_exc())
        send_telegram_alert(f"❌ Zerodha stocks data: unexpected error in data fetching session: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Zerodha Auto Login and 1-Minute Data Fetching for NSE Stocks')
    parser.add_argument('--token-only', action='store_true', 
                       help='Only get and update access token, do not fetch data')
    parser.add_argument('--from-date', type=str, 
                       help='Start date for data fetching (format: YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str, 
                       help='End date for data fetching (format: YYYY-MM-DD)')
    parser.add_argument('--skip-holiday-check', action='store_true',
                       help='Skip trading day check (for manual runs)')
    parser.add_argument('--nautilus-only', action='store_true',
                       help='Run only Nautilus transformation (for manual runs)')
    parser.add_argument('--date', type=str,
                       help='Date to process for Nautilus transformation (format: YYYY-MM-DD, defaults to today)')
    args = parser.parse_args()
    
    if args.nautilus_only:
        # Run only Nautilus transformation
        date_str = args.date if args.date else None
        run_nautilus_only(date_str=date_str)
        logger.info("Nautilus-only run completed. Exiting.")
        sys.exit(0)
    elif args.token_only:
        get_access_token_only()
    else:
        # Check if today is a trading day (unless skip flag is set)
        if not args.skip_holiday_check:
            if not is_trading_day():
                logger.info("Today is not a trading day (weekend or holiday). Exiting.")
                send_telegram_alert("ℹ️ Zerodha stocks data: today is not a trading day (weekend/holiday). Job skipped.")
                sys.exit(0)
            logger.info("Today is a trading day. Proceeding with data fetch.")
        
        # Initial alert that the daily job has started
        send_telegram_alert(
            f"🚀 Zerodha stocks data started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )
        
        from_date = None
        to_date = None
        
        if args.from_date:
            from_date = args.from_date
        if args.to_date:
            to_date = args.to_date
        
        # Run data fetching session (defaults to current day if dates not specified)
        run_data_fetching_session(
            from_date=from_date,
            to_date=to_date,
        )
        
        logger.info("Data fetching completed. Exiting.")
        # No continuous scheduler - Windows Task Scheduler handles daily execution
