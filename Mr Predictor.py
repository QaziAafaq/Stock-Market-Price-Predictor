"""
Mr. Predictor - Flask Web Trading Terminal
================================================
Install dependencies: pip install flask pandas numpy yfinance flask-cors
Run: python app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==============================================================================
# STOCK LIST
# ==============================================================================

STOCK_LIST = {
    "🔥 Popular": [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "MSFT", "name": "Microsoft"},
        {"ticker": "GOOGL", "name": "Google"},
        {"ticker": "AMZN", "name": "Amazon"},
        {"ticker": "TSLA", "name": "Tesla"},
        {"ticker": "NVDA", "name": "NVIDIA"},
        {"ticker": "META", "name": "Meta Platforms"},
        {"ticker": "NFLX", "name": "Netflix"}
    ],
    "💰 Crypto": [
        {"ticker": "BTC-USD", "name": "Bitcoin"},
        {"ticker": "ETH-USD", "name": "Ethereum"},
        {"ticker": "BNB-USD", "name": "Binance Coin"},
        {"ticker": "XRP-USD", "name": "Ripple"},
        {"ticker": "ADA-USD", "name": "Cardano"},
        {"ticker": "DOGE-USD", "name": "Dogecoin"},
        {"ticker": "SOL-USD", "name": "Solana"},
        {"ticker": "MATIC-USD", "name": "Polygon"}
    ],
    "🏦 Finance": [
        {"ticker": "JPM", "name": "JPMorgan Chase"},
        {"ticker": "BAC", "name": "Bank of America"},
        {"ticker": "WFC", "name": "Wells Fargo"},
        {"ticker": "GS", "name": "Goldman Sachs"},
        {"ticker": "V", "name": "Visa"},
        {"ticker": "MA", "name": "Mastercard"},
        {"ticker": "AXP", "name": "American Express"},
        {"ticker": "C", "name": "Citigroup"}
    ],
    "💊 Healthcare": [
        {"ticker": "JNJ", "name": "Johnson & Johnson"},
        {"ticker": "UNH", "name": "UnitedHealth"},
        {"ticker": "PFE", "name": "Pfizer"},
        {"ticker": "ABBV", "name": "AbbVie"},
        {"ticker": "TMO", "name": "Thermo Fisher"},
        {"ticker": "MRK", "name": "Merck"},
        {"ticker": "LLY", "name": "Eli Lilly"},
        {"ticker": "ABT", "name": "Abbott Labs"}
    ],
    "⚡ Tech": [
        {"ticker": "AMD", "name": "AMD"},
        {"ticker": "INTC", "name": "Intel"},
        {"ticker": "ORCL", "name": "Oracle"},
        {"ticker": "CSCO", "name": "Cisco"},
        {"ticker": "IBM", "name": "IBM"},
        {"ticker": "CRM", "name": "Salesforce"},
        {"ticker": "ADBE", "name": "Adobe"},
        {"ticker": "AVGO", "name": "Broadcom"}
    ],
    "🛍️ Consumer": [
        {"ticker": "WMT", "name": "Walmart"},
        {"ticker": "HD", "name": "Home Depot"},
        {"ticker": "DIS", "name": "Disney"},
        {"ticker": "NKE", "name": "Nike"},
        {"ticker": "MCD", "name": "McDonald's"},
        {"ticker": "SBUX", "name": "Starbucks"},
        {"ticker": "KO", "name": "Coca-Cola"},
        {"ticker": "PEP", "name": "PepsiCo"}
    ]
}

# ==============================================================================
# MARKET ENGINE
# ==============================================================================

class MarketEngine:
    def get_history(self, ticker, interval):
        period_map = {"1m": "1d", "1h": "1mo", "1d": "1y"}
        p = period_map.get(interval, "1y")
        
        try:
            df = yf.download(ticker, period=p, interval=interval, progress=False)
            if df.empty: return []
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            date_col = 'date' if 'date' in df.columns else 'datetime'
            
            data = []
            for _, row in df.iterrows():
                data.append({
                    "time": str(row[date_col]),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close'])
                })
            return data
        except Exception as e:
            print(f"History Error: {e}")
            return []

    def get_live_price(self, ticker):
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if df.empty: return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            return float(df['Close'].iloc[-1])
        except:
            return None

    def predict(self, history, current_price):
        if len(history) < 15 or current_price is None: return None
        
        window = history[-15:]
        weights = np.linspace(0.1, 1.0, len(window))
        changes = np.array([d['close'] - d['open'] for d in window])
        momentum = np.sum(changes * weights) / np.sum(weights)
        volatility = np.mean([d['high'] - d['low'] for d in window])
        
        start_point = current_price 
        move = momentum * 1.5
        max_move = volatility * 0.9
        
        if abs(move) > max_move:
            move = np.sign(move) * max_move
            
        pred_close = start_point + move
        
        return {
            "open": start_point,
            "close": pred_close,
            "high": max(start_point, pred_close) + (volatility * 0.2),
            "low": min(start_point, pred_close) - (volatility * 0.2),
            "is_prediction": True
        }

engine = MarketEngine()

# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stocks')
def get_stocks():
    return jsonify(STOCK_LIST)

@app.route('/api/data')
def get_data():
    ticker = request.args.get('ticker', 'BTC-USD')
    interval = request.args.get('interval', '1h')
    
    history = engine.get_history(ticker, interval)
    live_price = engine.get_live_price(ticker)
    prediction = engine.predict(history, live_price)
    
    return jsonify({
        'history': history[-50:],
        'live_price': live_price,
        'prediction': prediction,
        'timestamp': datetime.now().isoformat()
    })

# ==============================================================================
# HTML TEMPLATE
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mr. Predictor • Live Trading Terminal</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0f1e 0%, #1a1f3a 100%);
            color: white;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .navbar {
            background: linear-gradient(135deg, #0f1420 0%, #1a1f35 100%);
            padding: 20px 40px;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
            position: relative;
        }

        .navbar::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        }

        .nav-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1600px;
            margin: 0 auto;
            flex-wrap: wrap;
            gap: 20px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo {
            background: linear-gradient(135deg, #1e293b, #334155);
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 28px;
            border: 2px solid #3b82f6;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        }

        .brand-text h1 {
            font-size: 22px;
            font-weight: bold;
            letter-spacing: 1px;
        }

        .brand-text p {
            font-size: 11px;
            color: #64748b;
            margin-top: 2px;
        }

        .brand-text span {
            color: #3b82f6;
        }

        .controls {
            display: flex;
            gap: 25px;
            align-items: center;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .control-label {
            font-size: 9px;
            font-weight: bold;
            color: #94a3b8;
            letter-spacing: 1px;
        }

        .ticker-display {
            background: #1e293b;
            border: 2px solid #3b82f6;
            border-radius: 8px;
            padding: 10px 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .ticker-display:hover {
            background: #334155;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }

        .ticker-text {
            font-family: 'Consolas', monospace;
            font-size: 14px;
            font-weight: bold;
            min-width: 80px;
        }

        select {
            background: #1e293b;
            border: 2px solid #3b82f6;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        select:hover {
            background: #334155;
        }

        .start-btn {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .start-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        }

        .start-btn.running {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #64748b;
            animation: pulse 2s infinite;
        }

        .status-dot.active {
            background: #10b981;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .kpi-dashboard {
            max-width: 1600px;
            margin: 30px auto;
            padding: 0 40px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .kpi-card {
            background: linear-gradient(135deg, #1e293b, #0f1420);
            border: 2px solid transparent;
            border-radius: 16px;
            padding: 25px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        }

        .kpi-card:hover {
            border-color: #3b82f6;
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
        }

        .kpi-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }

        .kpi-icon {
            font-size: 22px;
        }

        .kpi-title {
            font-size: 10px;
            font-weight: bold;
            color: #94a3b8;
            letter-spacing: 1px;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: bold;
            margin-top: 5px;
            font-family: 'Consolas', monospace;
        }

        .chart-section {
            max-width: 1600px;
            margin: 0 auto 40px;
            padding: 0 40px;
        }

        .chart-container {
            background: linear-gradient(135deg, #0f1420, #1a1f35);
            border: 2px solid #3b82f6;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
            height: 600px;
        }

        #chart {
            width: 100%;
            height: 100%;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            backdrop-filter: blur(5px);
        }

        .modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: linear-gradient(135deg, #0f1420, #1a1f35);
            border: 2px solid #3b82f6;
            border-radius: 16px;
            width: 90%;
            max-width: 700px;
            max-height: 80vh;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }

        .modal-header {
            background: #1e293b;
            padding: 25px;
            border-bottom: 2px solid #3b82f6;
        }

        .modal-header h2 {
            font-size: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .search-box {
            margin-top: 15px;
            position: relative;
        }

        .search-box input {
            width: 100%;
            background: #0f1420;
            border: 2px solid #334155;
            color: white;
            padding: 12px 40px 12px 15px;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .search-box input:focus {
            outline: none;
            border-color: #3b82f6;
        }

        .search-icon {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 18px;
            color: #64748b;
        }

        .stocks-list {
            padding: 20px;
            max-height: 500px;
            overflow-y: auto;
        }

        .category {
            margin-bottom: 25px;
        }

        .category-header {
            font-size: 13px;
            font-weight: bold;
            color: #3b82f6;
            margin-bottom: 10px;
            padding: 10px 15px;
            background: #1e293b;
            border-radius: 8px;
        }

        .stock-item {
            background: #1e293b;
            padding: 15px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .stock-item:hover {
            background: #334155;
            border-color: #3b82f6;
            transform: translateX(5px);
        }

        .stock-ticker {
            font-weight: bold;
            color: #3b82f6;
            font-family: 'Consolas', monospace;
        }

        .stock-name {
            color: #94a3b8;
            font-size: 13px;
            margin-left: 10px;
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1e293b;
        }

        ::-webkit-scrollbar-thumb {
            background: #3b82f6;
            border-radius: 4px;
        }

        @media (max-width: 768px) {
            .nav-content {
                flex-direction: column;
            }

            .kpi-dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-content">
            <div class="brand">
                <div class="logo">🔮</div>
                <div class="brand-text">
                    <h1>MR. <span>PREDICTOR</span></h1>
                    <p>Professional Trading Terminal</p>
                </div>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label class="control-label">ASSET</label>
                    <div class="ticker-display" onclick="openStockSelector()">
                        <span class="ticker-text" id="selectedTicker">BTC-USD</span>
                        <span>⋮</span>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">TIMEFRAME</label>
                    <select id="timeframe">
                        <option value="1m">1 Minute</option>
                        <option value="1h" selected>1 Hour</option>
                        <option value="1d">1 Day</option>
                    </select>
                </div>
                
                <button class="start-btn" id="toggleBtn" onclick="toggleEngine()">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="btnText">⚡ START ENGINE</span>
                </button>
            </div>
        </div>
    </nav>

    <div class="kpi-dashboard">
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon">💵</div>
                <div class="kpi-title">LIVE MARKET PRICE</div>
            </div>
            <div class="kpi-value" id="livePrice">---</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-title">AI FORECAST TARGET</div>
            </div>
            <div class="kpi-value" style="color: #38bdf8;" id="forecastPrice">---</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon">📊</div>
                <div class="kpi-title">EXPECTED MOVEMENT</div>
            </div>
            <div class="kpi-value" id="movement">---</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-icon">🤖</div>
                <div class="kpi-title">TRADING SIGNAL</div>
            </div>
            <div class="kpi-value" id="signal">---</div>
        </div>
    </div>

    <div class="chart-section">
        <div class="chart-container">
            <div id="chart"></div>
        </div>
    </div>

    <div class="modal" id="stockModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>📊 Select Your Asset</h2>
                <div class="search-box">
                    <input type="text" id="searchInput" placeholder="Search stocks..." oninput="filterStocks()">
                    <span class="search-icon">🔍</span>
                </div>
            </div>
            <div class="stocks-list" id="stocksList"></div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let updateInterval = null;
        let chart = null;
        let candlestickSeries = null;
        let stocksData = {};

        // Initialize Lightweight Charts
        function initChart() {
            const chartContainer = document.getElementById('chart');
            chart = LightweightCharts.createChart(chartContainer, {
                layout: {
                    background: { color: '#0f1420' },
                    textColor: '#94a3b8',
                },
                grid: {
                    vertLines: { color: '#1e293b' },
                    horzLines: { color: '#1e293b' },
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                },
                rightPriceScale: {
                    borderColor: '#3b82f6',
                },
                timeScale: {
                    borderColor: '#3b82f6',
                    timeVisible: true,
                    secondsVisible: false,
                },
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
            });

            candlestickSeries = chart.addCandlestickSeries({
                upColor: '#10b981',
                downColor: '#ef4444',
                borderUpColor: '#10b981',
                borderDownColor: '#ef4444',
                wickUpColor: '#10b981',
                wickDownColor: '#ef4444',
            });

            // Handle window resize
            window.addEventListener('resize', () => {
                chart.applyOptions({
                    width: chartContainer.clientWidth,
                    height: chartContainer.clientHeight,
                });
            });
        }

        initChart();

        fetch('/api/stocks')
            .then(r => r.json())
            .then(data => {
                stocksData = data;
                renderStocksList();
            });

        function renderStocksList(filter = '') {
            const container = document.getElementById('stocksList');
            container.innerHTML = '';
            
            for (const [category, stocks] of Object.entries(stocksData)) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'category';
                
                const header = document.createElement('div');
                header.className = 'category-header';
                header.textContent = category;
                categoryDiv.appendChild(header);
                
                stocks.forEach(stock => {
                    const fullText = `${stock.ticker} ${stock.name}`.toLowerCase();
                    if (filter === '' || fullText.includes(filter.toLowerCase())) {
                        const item = document.createElement('div');
                        item.className = 'stock-item';
                        item.innerHTML = `<span class="stock-ticker">${stock.ticker}</span><span class="stock-name">${stock.name}</span>`;
                        item.onclick = () => selectStock(stock.ticker);
                        categoryDiv.appendChild(item);
                    }
                });
                
                if (categoryDiv.children.length > 1) {
                    container.appendChild(categoryDiv);
                }
            }
        }

        function filterStocks() {
            const filter = document.getElementById('searchInput').value;
            renderStocksList(filter);
        }

        function openStockSelector() {
            document.getElementById('stockModal').classList.add('active');
        }

        function selectStock(ticker) {
            document.getElementById('selectedTicker').textContent = ticker;
            document.getElementById('stockModal').classList.remove('active');
            if (isRunning) {
                updateData();
            }
        }

        document.getElementById('stockModal').onclick = function(e) {
            if (e.target === this) {
                this.classList.remove('active');
            }
        };

        function toggleEngine() {
            isRunning = !isRunning;
            const btn = document.getElementById('toggleBtn');
            const btnText = document.getElementById('btnText');
            const dot = document.getElementById('statusDot');
            
            if (isRunning) {
                btn.classList.add('running');
                btnText.textContent = '⏹ STOP ENGINE';
                dot.classList.add('active');
                updateData();
                updateInterval = setInterval(updateData, 5000);
            } else {
                btn.classList.remove('running');
                btnText.textContent = '⚡ START ENGINE';
                dot.classList.remove('active');
                clearInterval(updateInterval);
            }
        }

        function updateData() {
            const ticker = document.getElementById('selectedTicker').textContent;
            const interval = document.getElementById('timeframe').value;
            
            fetch(`/api/data?ticker=${ticker}&interval=${interval}`)
                .then(r => r.json())
                .then(data => {
                    updateKPIs(data);
                    updateChart(data);
                })
                .catch(err => console.error('Error:', err));
        }

        function updateKPIs(data) {
            const livePrice = data.live_price;
            const prediction = data.prediction;
            
            if (livePrice) {
                document.getElementById('livePrice').textContent = '$' + livePrice.toFixed(2);
            }
            
            if (prediction) {
                const target = prediction.close;
                const delta = target - livePrice;
                
                document.getElementById('forecastPrice').textContent = '$' + target.toFixed(2);
                
                const movementEl = document.getElementById('movement');
                const symbol = delta >= 0 ? '▲' : '▼';
                const color = delta >= 0 ? '#10b981' : '#ef4444';
                movementEl.textContent = `${symbol} ${Math.abs(delta).toFixed(2)}`;
                movementEl.style.color = color;
                
                const threshold = livePrice * 0.0002;
                const signalEl = document.getElementById('signal');
                if (Math.abs(delta) < threshold) {
                    signalEl.textContent = 'NEUTRAL';
                    signalEl.style.color = '#94a3b8';
                } else {
                    signalEl.textContent = delta > 0 ? '🟢 BUY' : '🔴 SELL';
                    signalEl.style.color = color;
                }
            }
        }

        function updateChart(data) {
            const candleData = data.history.map(d => ({
                time: Math.floor(new Date(d.time).getTime() / 1000),
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close
            }));
            
            // Add prediction candle in different color
            if (data.prediction) {
                const predTime = Math.floor(Date.now() / 1000) + 3600; // 1 hour ahead
                candleData.push({
                    time: predTime,
                    open: data.prediction.open,
                    high: data.prediction.high,
                    low: data.prediction.low,
                    close: data.prediction.close
                });
            }
            
            candlestickSeries.setData(candleData);
            chart.timeScale().fitContent();
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("🚀 Mr. Predictor starting...")
    print("📊 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)