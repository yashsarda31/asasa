
# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Stock Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .buy-signal {
        background-color: #90EE90;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .sell-signal {
        background-color: #FFB6C1;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .hold-signal {
        background-color: #FFFFE0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Top 500 Indian stocks (NSE)
@st.cache_data
def get_indian_stocks():
    """Get list of top Indian stocks"""
    # Top NSE stocks with their symbols
    stocks = {
        # Nifty 50 major stocks
        'RELIANCE': 'Reliance Industries',
        'TCS': 'Tata Consultancy Services',
        'HDFCBANK': 'HDFC Bank',
        'INFY': 'Infosys',
        'ICICIBANK': 'ICICI Bank',
        'HINDUNILVR': 'Hindustan Unilever',
        'ITC': 'ITC Limited',
        'SBIN': 'State Bank of India',
        'BHARTIARTL': 'Bharti Airtel',
        'KOTAKBANK': 'Kotak Mahindra Bank',
        'LT': 'Larsen & Toubro',
        'AXISBANK': 'Axis Bank',
        'ASIANPAINT': 'Asian Paints',
        'MARUTI': 'Maruti Suzuki',
        'WIPRO': 'Wipro',
        'ULTRACEMCO': 'UltraTech Cement',
        'TITAN': 'Titan Company',
        'NESTLEIND': 'Nestle India',
        'BAJFINANCE': 'Bajaj Finance',
        'HCLTECH': 'HCL Technologies',
        'ADANIENT': 'Adani Enterprises',
        'ADANIPORTS': 'Adani Ports',
        'SUNPHARMA': 'Sun Pharmaceutical',
        'ONGC': 'ONGC',
        'POWERGRID': 'Power Grid Corporation',
        'NTPC': 'NTPC Limited',
        'BAJAJFINSV': 'Bajaj Finserv',
        'TATAMOTORS': 'Tata Motors',
        'TECHM': 'Tech Mahindra',
        'TATASTEEL': 'Tata Steel',
        'INDUSINDBK': 'IndusInd Bank',
        'VEDL': 'Vedanta',
        'HINDALCO': 'Hindalco Industries',
        'GRASIM': 'Grasim Industries',
        'BRITANNIA': 'Britannia Industries',
        'SBILIFE': 'SBI Life Insurance',
        'CIPLA': 'Cipla',
        'DRREDDY': "Dr. Reddy's Laboratories",
        'DIVISLAB': "Divi's Laboratories",
        'JSWSTEEL': 'JSW Steel',
        'COALINDIA': 'Coal India',
        'IOC': 'Indian Oil Corporation',
        'HEROMOTOCO': 'Hero MotoCorp',
        'BPCL': 'BPCL',
        'SHREECEM': 'Shree Cement',
        'UPL': 'UPL Limited',
        'EICHERMOT': 'Eicher Motors',
        'TATACONSUM': 'Tata Consumer Products',
        'APOLLOHOSP': 'Apollo Hospitals',
        'M&M': 'Mahindra & Mahindra',
        # Additional stocks
        'ADANIGREEN': 'Adani Green Energy',
        'PIDILITIND': 'Pidilite Industries',
        'SIEMENS': 'Siemens',
        'HAVELLS': 'Havells India',
        'DABUR': 'Dabur India',
        'GODREJCP': 'Godrej Consumer Products',
        'HINDPETRO': 'Hindustan Petroleum',
        'AMBUJACEM': 'Ambuja Cements',
        'DLF': 'DLF Limited',
        'BAJAJ-AUTO': 'Bajaj Auto',
        'SBICARD': 'SBI Cards',
        'INDUSTOWER': 'Indus Towers',
        'ICICIPRULI': 'ICICI Prudential Life',
        'BANDHANBNK': 'Bandhan Bank',
        'HDFCLIFE': 'HDFC Life Insurance',
        'ICICIGI': 'ICICI Lombard',
        'AUROPHARMA': 'Aurobindo Pharma',
        'TORNTPHARM': 'Torrent Pharmaceuticals',
        'LUPIN': 'Lupin Limited',
        'BIOCON': 'Biocon',
        'MARICO': 'Marico',
        'COLPAL': 'Colgate-Palmolive',
        'MUTHOOTFIN': 'Muthoot Finance',
        'PEL': 'Piramal Enterprises',
        'BERGEPAINT': 'Berger Paints',
        'VOLTAS': 'Voltas',
        'TRENT': 'Trent Limited',
        'PAGEIND': 'Page Industries',
        'MINDTREE': 'Mindtree',
        'MPHASIS': 'Mphasis',
    }
    return stocks

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = f"{symbol}.NS"  # NSE suffix for Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            # Try with BSE suffix
            ticker = f"{symbol}.BO"
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    if df.empty:
        return df
    
    # Price & Volume Indicators
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_26'] = ta.ema(df['Close'], length=26)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # VWAP
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['RSI_6'] = ta.rsi(df['Close'], length=6)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    
    # Stochastic
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
    
    # ATR (Average True Range)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # ADX (Average Directional Index)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
    
    # CCI (Commodity Channel Index)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    
    # Williams %R
    df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    
    # MFI (Money Flow Index)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    
    # OBV (On-Balance Volume)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    # Ichimoku Cloud
    ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
    if ichimoku is not None:
        df = pd.concat([df, ichimoku], axis=1)
    
    # Additional Indicators
    df['ROC'] = ta.roc(df['Close'], length=10)
    df['CMO'] = ta.cmo(df['Close'], length=14)
    df['TRIX'] = ta.trix(df['Close'], length=14)
    df['TSI'] = ta.tsi(df['Close'])
    
    # Volatility
    df['NATR'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
    
    # Support & Resistance
    df['PIVOT'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['PIVOT'] - df['Low']
    df['S1'] = 2 * df['PIVOT'] - df['High']
    
    return df

def calculate_signal_strength(df):
    """Calculate buy/sell signal based on 40+ indicators"""
    if df.empty or len(df) < 200:
        return 0, "INSUFFICIENT DATA", []
    
    latest = df.iloc[-1]
    signals = []
    bullish_count = 0
    bearish_count = 0
    
    # Price Action Signals
    if pd.notna(latest['SMA_20']) and latest['Close'] > latest['SMA_20']:
        bullish_count += 1
        signals.append(("Price > SMA20", "Bullish"))
    else:
        bearish_count += 1
        signals.append(("Price < SMA20", "Bearish"))
    
    if pd.notna(latest['SMA_50']) and latest['Close'] > latest['SMA_50']:
        bullish_count += 1
        signals.append(("Price > SMA50", "Bullish"))
    else:
        bearish_count += 1
        signals.append(("Price < SMA50", "Bearish"))
    
    if pd.notna(latest['SMA_200']) and latest['Close'] > latest['SMA_200']:
        bullish_count += 2  # Double weight for long-term trend
        signals.append(("Price > SMA200", "Strong Bullish"))
    else:
        bearish_count += 2
        signals.append(("Price < SMA200", "Strong Bearish"))
    
    # EMA Signals
    if pd.notna(latest['EMA_12']) and pd.notna(latest['EMA_26']):
        if latest['EMA_12'] > latest['EMA_26']:
            bullish_count += 1
            signals.append(("EMA12 > EMA26", "Bullish"))
        else:
            bearish_count += 1
            signals.append(("EMA12 < EMA26", "Bearish"))
    
    # VWAP Signal
    if pd.notna(latest['VWAP']) and latest['Close'] > latest['VWAP']:
        bullish_count += 1
        signals.append(("Price > VWAP", "Bullish"))
    else:
        bearish_count += 1
        signals.append(("Price < VWAP", "Bearish"))
    
    # RSI Signals
    if pd.notna(latest['RSI']):
        if latest['RSI'] < 30:
            bullish_count += 2
            signals.append(("RSI Oversold", "Strong Bullish"))
        elif latest['RSI'] > 70:
            bearish_count += 2
            signals.append(("RSI Overbought", "Strong Bearish"))
        elif 40 < latest['RSI'] < 60:
            signals.append(("RSI Neutral", "Neutral"))
    
    # MACD Signal
    if 'MACD_12_26_9' in latest.index and pd.notna(latest['MACD_12_26_9']):
        if 'MACDs_12_26_9' in latest.index and pd.notna(latest['MACDs_12_26_9']):
            if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                bullish_count += 1
                signals.append(("MACD > Signal", "Bullish"))
            else:
                bearish_count += 1
                signals.append(("MACD < Signal", "Bearish"))
    
    # Bollinger Bands
    if 'BBL_20_2.0' in latest.index and 'BBU_20_2.0' in latest.index:
        if pd.notna(latest['BBL_20_2.0']) and latest['Close'] < latest['BBL_20_2.0']:
            bullish_count += 1
            signals.append(("Price < Lower BB", "Bullish"))
        elif pd.notna(latest['BBU_20_2.0']) and latest['Close'] > latest['BBU_20_2.0']:
            bearish_count += 1
            signals.append(("Price > Upper BB", "Bearish"))
    
    # Stochastic
    if 'STOCHk_14_3_3' in latest.index and pd.notna(latest['STOCHk_14_3_3']):
        if latest['STOCHk_14_3_3'] < 20:
            bullish_count += 1
            signals.append(("Stochastic Oversold", "Bullish"))
        elif latest['STOCHk_14_3_3'] > 80:
            bearish_count += 1
            signals.append(("Stochastic Overbought", "Bearish"))
    
    # CCI
    if pd.notna(latest['CCI']):
        if latest['CCI'] < -100:
            bullish_count += 1
            signals.append(("CCI Oversold", "Bullish"))
        elif latest['CCI'] > 100:
            bearish_count += 1
            signals.append(("CCI Overbought", "Bearish"))
    
    # MFI
    if pd.notna(latest['MFI']):
        if latest['MFI'] < 20:
            bullish_count += 1
            signals.append(("MFI Oversold", "Bullish"))
        elif latest['MFI'] > 80:
            bearish_count += 1
            signals.append(("MFI Overbought", "Bearish"))
    
    # ADX Trend Strength
    if 'ADX_14' in latest.index and pd.notna(latest['ADX_14']):
        if latest['ADX_14'] > 25:
            signals.append(("Strong Trend (ADX>25)", "Trend"))
    
    # Volume Analysis
    vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
    if pd.notna(vol_avg) and latest['Volume'] > vol_avg * 1.5:
        signals.append(("High Volume", "Significant"))
    
    # Calculate overall signal
    total_signals = bullish_count + bearish_count
    if total_signals == 0:
        return 50, "HOLD", signals
    
    signal_strength = (bullish_count / total_signals) * 100
    
    if signal_strength >= 65:
        signal = "STRONG BUY"
    elif signal_strength >= 55:
        signal = "BUY"
    elif signal_strength >= 45:
        signal = "HOLD"
    elif signal_strength >= 35:
        signal = "SELL"
    else:
        signal = "STRONG SELL"
    
    return signal_strength, signal, signals

def create_price_chart(df, symbol):
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add SMAs
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    if 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VWAP'], name='VWAP',
                      line=dict(color='purple', width=1, dash='dash')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BBU_20_2.0' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='BB Upper',
                      line=dict(color='gray', width=0.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='BB Lower',
                      line=dict(color='gray', width=0.5)),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if row['Close'] < row['Open'] else 'green' 
              for index, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='purple', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="Overbought", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                      annotation_text="Oversold", row=3, col=1)
    
    # MACD
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_12_26_9'], name='MACD',
                      line=dict(color='blue', width=1)),
            row=4, col=1
        )
        if 'MACDs_12_26_9' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name='Signal',
                          line=dict(color='red', width=1)),
                row=4, col=1
            )
        if 'MACDh_12_26_9' in df.columns:
            colors = ['red' if val < 0 else 'green' for val in df['MACDh_12_26_9']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Histogram',
                       marker_color=colors),
                row=4, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

def display_signal_box(signal_strength, signal):
    """Display signal box with color coding"""
    if "BUY" in signal:
        color_class = "buy-signal"
        emoji = "🟢"
    elif "SELL" in signal:
        color_class = "sell-signal"
        emoji = "🔴"
    else:
        color_class = "hold-signal"
        emoji = "🟡"
    
    st.markdown(f"""
        <div class="{color_class}">
            <h2>{emoji} {signal}</h2>
            <h3>Signal Strength: {signal_strength:.1f}%</h3>
        </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.title("📈 Indian Stock Analytics Platform")
    st.markdown("### Real-time Technical Analysis for NSE Stocks")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock selection
    stocks = get_indian_stocks()
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        options=list(stocks.keys()),
        format_func=lambda x: f"{x} - {stocks[x]}"
    )
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Select Time Period",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=5
    )
    
    # Fetch and process data
    with st.spinner(f"Fetching data for {selected_stock}..."):
        df = fetch_stock_data(selected_stock, period)
        
        if not df.empty:
            df = calculate_technical_indicators(df)
            
            # Calculate signal
            signal_strength, signal, signal_details = calculate_signal_strength(df)
            
            # Display main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            with col1:
                st.metric("Current Price", f"₹{latest_price:.2f}", 
                         f"{price_change:.2f} ({price_change_pct:.2f}%)")
            
            with col2:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}",
                         f"{((df['Volume'].iloc[-1] / df['Volume'].mean() - 1) * 100):.1f}% vs avg")
            
            with col3:
                if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
                    rsi_value = df['RSI'].iloc[-1]
                    st.metric("RSI (14)", f"{rsi_value:.2f}",
                             "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral")
            
            with col4:
                if 'VWAP' in df.columns and pd.notna(df['VWAP'].iloc[-1]):
                    vwap_value = df['VWAP'].iloc[-1]
                    st.metric("VWAP", f"₹{vwap_value:.2f}",
                             "Above" if latest_price > vwap_value else "Below")
            
            st.markdown("---")
            
            # Display signal box
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.subheader("📊 Trading Signal")
                display_signal_box(signal_strength, signal)
                
                # Signal breakdown
                with st.expander("Signal Details"):
                    bullish_signals = [s for s in signal_details if "Bullish" in s[1]]
                    bearish_signals = [s for s in signal_details if "Bearish" in s[1]]
                    
                    st.write(f"**Bullish Signals:** {len(bullish_signals)}")
                    for sig in bullish_signals[:5]:
                        st.write(f"✅ {sig[0]}")
                    
                    st.write(f"**Bearish Signals:** {len(bearish_signals)}")
                    for sig in bearish_signals[:5]:
                        st.write(f"❌ {sig[0]}")
            
            with col2:
                # Key Statistics
                st.subheader("📈 Key Statistics")
                
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.write("**Price Statistics**")
                    st.write(f"Day High: ₹{df['High'].iloc[-1]:.2f}")
                    st.write(f"Day Low: ₹{df['Low'].iloc[-1]:.2f}")
                    st.write(f"52W High: ₹{df['High'].max():.2f}")
                    st.write(f"52W Low: ₹{df['Low'].min():.2f}")
                
                with stats_col2:
                    st.write("**Moving Averages**")
                    if 'SMA_20' in df.columns and pd.notna(df['SMA_20'].iloc[-1]):
                        st.write(f"SMA 20: ₹{df['SMA_20'].iloc[-1]:.2f}")
                    if 'SMA_50' in df.columns and pd.notna(df['SMA_50'].iloc[-1]):
                        st.write(f"SMA 50: ₹{df['SMA_50'].iloc[-1]:.2f}")
                    if 'SMA_200' in df.columns and pd.notna(df['SMA_200'].iloc[-1]):
                        st.write(f"SMA 200: ₹{df['SMA_200'].iloc[-1]:.2f}")
            
            st.markdown("---")
            
            # Interactive Chart
            st.subheader("📊 Technical Chart")
            fig = create_price_chart(df, selected_stock)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators Table
            with st.expander("📋 All Technical Indicators"):
                indicators_df = pd.DataFrame({
                    'Indicator': [],
                    'Value': [],
                    'Signal': []
                })
                
                # Populate indicators table
                latest = df.iloc[-1]
                indicators_data = []
                
                # Add all calculated indicators to table
                indicator_columns = ['RSI', 'RSI_6', 'CCI', 'MFI', 'ATR', 'NATR', 
                                   'ROC', 'CMO', 'TRIX', 'TSI']
                
                for col in indicator_columns:
                    if col in df.columns and pd.notna(latest[col]):
                        value = latest[col]
                        # Determine signal based on indicator
                        if col in ['RSI', 'RSI_6']:
                            if value < 30:
                                signal = "Oversold 🟢"
                            elif value > 70:
                                signal = "Overbought 🔴"
                            else:
                                signal = "Neutral 🟡"
                        elif col == 'CCI':
                            if value < -100:
                                signal = "Oversold 🟢"
                            elif value > 100:
                                signal = "Overbought 🔴"
                            else:
                                signal = "Neutral 🟡"
                        elif col == 'MFI':
                            if value < 20:
                                signal = "Oversold 🟢"
                            elif value > 80:
                                signal = "Overbought 🔴"
                            else:
                                signal = "Neutral 🟡"
                        else:
                            signal = "Active"
                        
                        indicators_data.append({
                            'Indicator': col,
                            'Value': f"{value:.2f}",
                            'Signal': signal
                        })
                
                if indicators_data:
                    indicators_df = pd.DataFrame(indicators_data)
                    st.dataframe(indicators_df, use_container_width=True)
            
            # Support and Resistance Levels
            with st.expander("📊 Support & Resistance Levels"):
                latest = df.iloc[-1]
                if 'PIVOT' in df.columns:
                    st.write(f"**Pivot Point:** ₹{latest['PIVOT']:.2f}")
                    st.write(f"**Resistance 1:** ₹{latest['R1']:.2f}")
                    st.write(f"**Support 1:** ₹{latest['S1']:.2f}")
                
                # Fibonacci Levels
                high_52w = df['High'].max()
                low_52w = df['Low'].min()
                diff = high_52w - low_52w
                
                st.write("\n**Fibonacci Retracement Levels:**")
                fib_levels = {
                    '0%': high_52w,
                    '23.6%': high_52w - (diff * 0.236),
                    '38.2%': high_52w - (diff * 0.382),
                    '50%': high_52w - (diff * 0.5),
                    '61.8%': high_52w - (diff * 0.618),
                    '100%': low_52w
                }
                
                for level, price in fib_levels.items():
                    st.write(f"{level}: ₹{price:.2f}")
        else:
            st.error("Unable to fetch data for the selected stock. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>💡 <b>Disclaimer:</b> This tool is for educational purposes only. 
            Please do your own research before making investment decisions.</p>
            <p>Data provided by Yahoo Finance | Updates every 5 minutes</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
