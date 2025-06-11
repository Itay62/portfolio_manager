import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
import json # For saving and loading data
import os   # For checking file existence

# --- Configuration ---
APP_TITLE = "My Portfolio Manager"
REFRESH_INTERVAL_SECONDS = 300
DATA_FILE = "portfolio_data.json" # File to store portfolio data

# --- Helper Functions ---
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def get_current_price(ticker_symbol):
    # ... (same as before)
    try:
        ticker = yf.Ticker(ticker_symbol)
        todays_data = ticker.history(period='1d')
        if not todays_data.empty:
            return todays_data['Close'].iloc[-1]
        else:
            info = ticker.info
            if 'regularMarketPreviousClose' in info:
                return info['regularMarketPreviousClose']
            st.warning(f"Could not fetch current price for {ticker_symbol}. Using 0.")
            return 0.0
    except Exception as e:
        st.error(f"Error fetching price for {ticker_symbol}: {e}")
        return 0.0

def save_portfolio_data():
    """Saves the current portfolio data to a JSON file."""
    data_to_save = {
        'initial_cash': st.session_state.initial_cash,
        'cash_flows': st.session_state.cash_flows,
        'transactions': st.session_state.transactions
    }
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        st.sidebar.success(f"Data saved to {DATA_FILE}")
    except Exception as e:
        st.sidebar.error(f"Error saving data: {e}")

def load_portfolio_data():
    """Loads portfolio data from the JSON file if it exists."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                st.session_state.initial_cash = data.get('initial_cash', 0.0)
                st.session_state.cash_flows = data.get('cash_flows', [])
                st.session_state.transactions = data.get('transactions', [])
                # Ensure dates in loaded data are strings if they were saved as such
                # (json automatically handles this for simple date strings)
                return True # Data loaded
        except json.JSONDecodeError:
            st.sidebar.error(f"Error decoding {DATA_FILE}. File might be corrupted. Starting fresh.")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}. Starting fresh.")
    return False # No data loaded or error

def initialize_session_state():
    """Initializes session state variables, loading from file if available."""
    if 'initialized' not in st.session_state: # Run only once per session
        data_loaded = load_portfolio_data()
        if not data_loaded: # If no data was loaded, set defaults
            if 'initial_cash' not in st.session_state:
                st.session_state.initial_cash = 0.0
            if 'cash_flows' not in st.session_state:
                st.session_state.cash_flows = []
            if 'transactions' not in st.session_state:
                st.session_state.transactions = []

        # This is derived, so initialize empty if not already part of loading
        if 'portfolio_summary' not in st.session_state:
            st.session_state.portfolio_summary = {
                "holdings": {},
                "total_portfolio_value": 0.0,
                "total_profit_loss": 0.0,
                "current_cash": 0.0,
                "net_capital_invested": 0.0
            }
        st.session_state.initialized = True # Mark as initialized

def calculate_portfolio_summary():
    # ... (same as before)
    current_cash = st.session_state.initial_cash
    net_capital_invested = st.session_state.initial_cash

    for cf in st.session_state.cash_flows:
        current_cash += cf['amount']
        net_capital_invested += cf['amount']

    holdings = {}
    realized_profit_loss = 0.0  # Track realized P/L
    buy_records = {}  # Track buy prices per ticker

    for t in st.session_state.transactions:
        cost = t['quantity'] * t['buy_price']
        current_cash -= cost

        if t['quantity'] > 0:  # Buy
            if t['ticker'] not in buy_records:
                buy_records[t['ticker']] = {'quantity': 0, 'total_cost': 0}
            buy_records[t['ticker']]['quantity'] += t['quantity']
            buy_records[t['ticker']]['total_cost'] += t['quantity'] * t['buy_price']
        else:  # Sell
            if t['ticker'] in buy_records and buy_records[t['ticker']]['quantity'] > 0:
                avg_buy_price = buy_records[t['ticker']]['total_cost'] / buy_records[t['ticker']]['quantity']
                realized_pl = (-t['quantity']) * (t['buy_price'] - avg_buy_price)
                realized_profit_loss += realized_pl

        if t['ticker'] not in holdings:
            holdings[t['ticker']] = {'quantity': 0, 'total_cost': 0.0}
        holdings[t['ticker']]['quantity'] += t['quantity']
        holdings[t['ticker']]['total_cost'] += cost

    total_assets_value = 0.0
    detailed_holdings = {}

    for ticker, data in holdings.items():
        if data['quantity'] > 0:
            current_price = get_current_price(ticker)
            current_value = data['quantity'] * current_price
            avg_buy_price = data['total_cost'] / data['quantity']
            profit_loss_holding = current_value - data['total_cost']

            detailed_holdings[ticker] = {
                'quantity': data['quantity'],
                'avg_buy_price': avg_buy_price,
                'current_price': current_price,
                'current_value': current_value,
                'profit_loss': profit_loss_holding
            }
            total_assets_value += current_value

    total_portfolio_value = total_assets_value + current_cash
    total_profit_loss = total_portfolio_value - net_capital_invested

    st.session_state.portfolio_summary = {
        "holdings": detailed_holdings,
        "total_portfolio_value": total_portfolio_value,
        "total_profit_loss": total_profit_loss,
        "realized_profit_loss": realized_profit_loss,  # Add realized P/L
        "current_cash": current_cash,
        "net_capital_invested": net_capital_invested
    }


# --- Streamlit App UI ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Initialize session state (this will also attempt to load data)
initialize_session_state()

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Settings & Inputs")

    # Save Data Button
    if st.button("💾 Save Portfolio Data"):
        save_portfolio_data()
        # No rerun needed here, just a save action

    st.markdown("---") # Visual separator

    # Initial Cash
    st.subheader("💵 Initial Cash")
    # Use a key for number_input to ensure it reflects session_state correctly after load
    new_initial_cash = st.number_input(
        "Set Initial Cash",
        min_value=0.0,
        value=float(st.session_state.initial_cash), # Ensure it's float
        step=100.0,
        format="%.2f",
        key="initial_cash_input" # Unique key
    )
    if new_initial_cash != st.session_state.initial_cash:
        st.session_state.initial_cash = new_initial_cash
        save_portfolio_data() # Optionally auto-save on change
        st.rerun()

    # Cash Flow
    st.subheader("💸 Cash Flow (Deposits/Withdrawals)")
    with st.form("cash_flow_form", clear_on_submit=True):
        cf_amount = st.number_input("Amount (+ for deposit, - for withdrawal)", step=50.0, format="%.2f")
        cf_description = st.text_input("Description (e.g., Monthly Savings)")
        cf_date = st.date_input("Date", datetime.today())
        cf_submitted = st.form_submit_button("Add Cash Flow")

        if cf_submitted and cf_amount != 0:
            st.session_state.cash_flows.append({
                'amount': cf_amount,
                'date': cf_date.strftime("%Y-%m-%d"),
                'description': cf_description
            })
            st.success(f"Cash flow of {cf_amount:,.2f} added.")
            save_portfolio_data() # Auto-save after adding
            st.rerun()

    # New Transaction Entry
    st.subheader("📊 Add New Buy Transaction")
    with st.form("transaction_form", clear_on_submit=True):
        ticker = st.text_input("Ticker Symbol (e.g., AAPL)").upper()
        # MODIFIED: Replaced quantity input with total_investment input
        total_investment = st.number_input(
            "Total Investment Amount",
            min_value=0.01,
            step=1.00,
            format="%.2f",
            help="The total amount of money you want to invest in this ticker for this transaction."
        )
        buy_price = st.number_input(
            "Buy Price per Share",
            min_value=0.01,
            step=0.01,
            format="%.2f",
            help="The price per share at the time of purchase."
        )
        buy_date = st.date_input("Buy Date", datetime.today())
        submitted = st.form_submit_button("Add Buy Transaction")

        # MODIFIED: Condition and logic to handle new inputs
        if submitted and ticker and total_investment > 0 and buy_price > 0:
            calculated_quantity = total_investment / buy_price
            st.session_state.transactions.append({
                'ticker': ticker,
                'quantity': calculated_quantity, # Store calculated quantity
                'buy_price': buy_price,
                'date': buy_date.strftime("%Y-%m-%d")
            })
            # MODIFIED: Success message
            st.success(f"Recorded investment of ${total_investment:,.2f} in {ticker} at ${buy_price:,.2f} per share (approx. {calculated_quantity:,.4f} shares).")
            save_portfolio_data() # Auto-save after adding
            st.rerun()
        elif submitted:
            # MODIFIED: More specific error message
            st.error("Please fill in all fields correctly (Ticker, Total Investment > 0, Buy Price > 0).")

    if st.button("🔄 Refresh Prices & Recalculate"):
        st.cache_data.clear()
        st.rerun()

        # Add this after the buy transaction form in the sidebar
    st.subheader("📉 Sell Position")
    with st.form("sell_transaction_form", clear_on_submit=True):
        # Make sure portfolio summary is calculated before accessing holdings
        calculate_portfolio_summary()
        summary = st.session_state.portfolio_summary

        # Get current holdings for dropdown with their buy prices
        current_holdings = {}
        for ticker, data in summary['holdings'].items():
            if data['quantity'] > 0:
                # Find original buy price from transactions
                buy_transactions = [t for t in st.session_state.transactions
                                    if t['ticker'] == ticker and t['quantity'] > 0]
                if buy_transactions:
                    avg_buy_price = sum(t['buy_price'] * t['quantity'] for t in buy_transactions) / \
                                    sum(t['quantity'] for t in buy_transactions)
                    current_holdings[ticker] = {
                        'quantity': data['quantity'],
                        'avg_buy_price': avg_buy_price
                    }

        if current_holdings:
            # Dropdown for selecting ticker from current holdings
            sell_ticker = st.selectbox(
                "Select Position to Sell",
                options=list(current_holdings.keys()),
                format_func=lambda
                    x: f"{x} ({current_holdings[x]['quantity']:,.4f} shares, bought @ ${current_holdings[x]['avg_buy_price']:,.2f})"
            )

            # Get position details
            position = current_holdings[sell_ticker]
            max_shares = position['quantity']
            buy_price = position['avg_buy_price']

            # Allow manual sell price entry
            sell_price = st.number_input(
                "Sell Price ($)",
                min_value=0.01,
                value=float(buy_price),  # Default to buy price instead of current price
                step=0.01,
                format="%.2f",
                help=f"Original buy price: ${buy_price:,.2f}"
            )

            # By default, set to sell entire position
            st.write(f"Selling entire position: {max_shares:,.4f} shares")
            sell_quantity = max_shares
            sell_amount = sell_quantity * sell_price

            # Calculate and display preview with profit/loss
            total_cost = sell_quantity * buy_price
            profit_loss = sell_amount - total_cost

            st.write("Transaction Preview:")
            st.write(f"Original Cost: ${total_cost:,.2f} (${buy_price:,.2f}/share)")
            st.write(f"Sell Proceeds: ${sell_amount:,.2f} (${sell_price:,.2f}/share)")
            if profit_loss > 0:
                st.success(f"Realized Profit: ${profit_loss:,.2f}")
            else:
                st.error(f"Realized Loss: ${profit_loss:,.2f}")

            sell_submitted = st.form_submit_button("Execute Sell")

            if sell_submitted:
                # Add sell transaction (negative quantity)
                st.session_state.transactions.append({
                    'ticker': sell_ticker,
                    'quantity': -sell_quantity,  # Negative for sell
                    'buy_price': sell_price,  # Using specified sell price
                    'date': datetime.today().strftime("%Y-%m-%d")
                })
                save_portfolio_data()  # Auto-save after adding
                message = (f"Sold {sell_quantity:,.4f} shares of {sell_ticker} for ${sell_amount:,.2f}\n"
                           f"Realized {'Profit' if profit_loss > 0 else 'Loss'}: ${abs(profit_loss):,.2f}")
                if profit_loss > 0:
                    st.success(message)
                else:
                    st.warning(message)
                st.rerun()
        else:
            st.info("No positions available to sell. Add some buy transactions first.")

# --- Main Display Area ---
calculate_portfolio_summary()
summary = st.session_state.portfolio_summary

st.header("📈 Portfolio Overview")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Portfolio Value", f"${summary['total_portfolio_value']:,.2f}")
col2.metric("Total Profit/Loss", f"${summary['total_profit_loss']:,.2f}",
            delta=f"{((summary['total_profit_loss']/summary['net_capital_invested'] if summary['net_capital_invested'] else 0)*100):,.2f}%" if summary['net_capital_invested'] !=0 else "N/A")
col3.metric("Lifetime Realized P/L", f"${summary['realized_profit_loss']:,.2f}")
col4.metric("Net Capital Invested", f"${summary['net_capital_invested']:,.2f}")
col5.metric("Current Cash", f"${summary['current_cash']:,.2f}")

st.markdown("---")

col_chart, col_holdings_table = st.columns([2, 3])

with col_chart:
    st.subheader("📊 Holdings Distribution")
    holdings_for_pie = []
    if summary['holdings']:
        # Use total_portfolio_value instead of just sum of holdings
        total_portfolio = summary['total_portfolio_value']  # This includes cash
        for ticker, data in summary['holdings'].items():
            if data.get('current_value', 0) > 0:  # Check if current_value exists and is > 0
                percentage = (data['current_value'] / total_portfolio * 100)
                holdings_for_pie.append({
                    'Ticker': f"{ticker} ({percentage:.1f}%)",
                    'Value': data['current_value'],
                    'Percentage': percentage
                })

        # Add cash as a segment in the pie chart
        cash_percentage = (summary['current_cash'] / total_portfolio * 100)
        holdings_for_pie.append({
            'Ticker': f"Cash ({cash_percentage:.1f}%)",
            'Value': summary['current_cash'],
            'Percentage': cash_percentage
        })

    if holdings_for_pie:
        pie_df = pd.DataFrame(holdings_for_pie)
        fig_pie = px.pie(
            pie_df,
            names='Ticker',
            values='Value',
            title='Portfolio Asset Allocation (% of Total Portfolio)'
        )
        # Customize the hover text to show value and percentage
        fig_pie.update_traces(
            textposition='inside',
            textinfo='label',  # This will show the ticker with percentage we added
            hovertemplate="<b>%{label}</b><br>" +
                          "Value: $%{value:,.2f}<br>" +
                          "<extra></extra>"  # This removes the secondary box in the hover
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No assets with positive value to display in pie chart.")

with col_holdings_table:
    st.subheader("📋 Current Holdings")
    if summary['holdings']:
        holdings_df_data = []
        for ticker, data in summary['holdings'].items():
            holdings_df_data.append({
                "Ticker": ticker,
                "% Change": f"{((data.get('current_price',0) - data['avg_buy_price']) / data['avg_buy_price'] * 100):,.2f}%" if data['avg_buy_price'] and data.get('current_price') is not None else "N/A",
                "P/L": f"${data.get('profit_loss', 0.0):,.2f}",
                "Current Value": f"${data.get('current_value', 0.0):,.2f}",
                "Avg Buy Price": f"${data['avg_buy_price']:,.2f}",
                "Current Price": f"${data.get('current_price', 0.0):,.2f}",
                "Quantity": data['quantity'], # Will show calculated (potentially fractional) quantity
            })
        holdings_df = pd.DataFrame(holdings_df_data)
        # Display quantity with more precision if desired, e.g., by formatting the column
        # For now, default Pandas formatting will apply.
        st.dataframe(holdings_df, use_container_width=True)
    else:
        st.info("No holdings yet. Add some transactions.")

st.markdown("---")

with st.expander("📜 Transaction History"):
    if st.session_state.transactions:
        trans_df = pd.DataFrame(st.session_state.transactions)
        if not trans_df.empty:
            trans_df['Total Cost'] = trans_df['quantity'] * trans_df['buy_price']
            # The 'Total Cost' here will now be very close to the 'Total Investment Amount' user entered.
            st.dataframe(trans_df[['date', 'ticker', 'quantity', 'buy_price', 'Total Cost']], use_container_width=True)
        else:
            st.write("No transactions recorded.")
    else:
        st.write("No transactions recorded.")

with st.expander("💵 Cash Flow History"):
    if st.session_state.cash_flows:
        cash_flow_df = pd.DataFrame(st.session_state.cash_flows)
        if not cash_flow_df.empty:
            st.dataframe(cash_flow_df[['date', 'amount', 'description']], use_container_width=True)
        else:
            st.write("No cash flow entries recorded.")
    else:
        st.write("No cash flow entries recorded.")