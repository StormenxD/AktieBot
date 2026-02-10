"""
Save Backtest Results for Vercel Deployment
============================================

This script runs your backtest and saves the results to a JSON file
that can be deployed to Vercel.

Usage:
    python save_backtest_for_vercel.py

Output:
    backtest_results.json - All results needed for dashboard
"""

import json
from datetime import datetime

def save_results_for_vercel(portfolio_series, buy_markers, sell_markers, 
                             partial_exit_markers, trades_df, current_positions, 
                             metrics, cash, signals, prices):
    """
    Save all backtest results to a JSON file for Vercel deployment
    """
    
    print("\n" + "="*60)
    print("💾 SAVING RESULTS FOR VERCEL DEPLOYMENT")
    print("="*60)
    
    # Convert datetime objects to ISO format strings
    def convert_markers(markers_dict):
        return {
            stock: [(date.isoformat(), float(z)) for date, z in markers]
            for stock, markers in markers_dict.items()
        }
    
    def convert_positions(pos_dict):
        result = {}
        for stock, pos in pos_dict.items():
            result[stock] = {
                'entry_date': pos['entry_date'].isoformat(),
                'entry_price': float(pos['entry_price']),
                'entry_z': float(pos['entry_z']),
                'current_price': float(pos['current_price']),
                'current_z': float(pos['current_z']),
                'pnl_pct': float(pos['pnl_pct']),
                'days_held': int(pos['days_held']),
                'partial': bool(pos['partial']),
                'shares': float(pos['shares']),
                'position_value': float(pos['position_value'])
            }
        return result
    
    # Build results dictionary
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'backtest_period': {
                'start': portfolio_series.index[0].isoformat(),
                'end': portfolio_series.index[-1].isoformat()
            },
            'parameters': {
                'WINDOW': 65,
                'ENTRY_Z': -1.7,
                'STOP_LOSS_PCT': 0.18,
                'START_CAPITAL': 100000
            }
        },
        
        'portfolio_series': {
            'dates': [d.isoformat() for d in portfolio_series.index],
            'values': [float(v) for v in portfolio_series.values]
        },
        
        'markers': {
            'buy': convert_markers(buy_markers),
            'sell': convert_markers(sell_markers),
            'partial_exit': convert_markers(partial_exit_markers)
        },
        
        'trades': [
            {
                'date': trade['date'].isoformat() if hasattr(trade['date'], 'isoformat') else str(trade['date']),
                'stock': trade['stock'],
                'action': trade['action'],
                'price': float(trade['price']),
                'shares': float(trade.get('shares', 0)),
                'z_score': float(trade.get('z_score', 0)),
                'profit_pct': float(trade.get('profit_pct', 0)) if 'profit_pct' in trade else None,
                'reason': trade.get('reason', '')
            }
            for trade in trades_df.to_dict('records')
        ],
        
        'current_positions': convert_positions(current_positions),
        
        'metrics': {
            k: str(v) for k, v in metrics.items()
        },
        
        'cash': float(cash),
        
        'signals': {
            'dates': [d.isoformat() for d in signals.index],
            'stocks': list(signals.columns),
            'values': signals.values.tolist()
        },
        
        'latest_prices': {
            stock: float(prices[stock].iloc[-1])
            for stock in prices.columns
            if stock in signals.columns
        }
    }
    
    # Save to JSON
    output_file = 'backtest_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"📦 File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 10:
        print("\n⚠️  WARNING: File is large (>10MB)")
        print("   Consider compressing with gzip:")
        print("   import gzip")
        print("   with gzip.open('backtest_results.json.gz', 'wt') as f:")
        print("       json.dump(results, f)")
    
    print("\n📋 Summary:")
    print(f"   Portfolio days: {len(portfolio_series)}")
    print(f"   Total trades: {len(trades_df)}")
    print(f"   Open positions: {len(current_positions)}")
    print(f"   Stocks tracked: {len(signals.columns)}")
    
    print("\n🚀 Next steps:")
    print("   1. Copy this file to your Vercel project folder")
    print("   2. Deploy to Vercel")
    print("   3. Your dashboard will load these pre-computed results")
    
    return output_file

# Example usage (add to end of your main script):
"""
# After running backtest, save results:
if __name__ == "__main__":
    # ... your backtest code ...
    
    # Save for Vercel
    save_results_for_vercel(
        portfolio_series=portfolio_series,
        buy_markers=buy_markers,
        sell_markers=sell_markers,
        partial_exit_markers=partial_exit_markers,
        trades_df=trades_df,
        current_positions=current_positions,
        metrics=metrics,
        cash=cash,
        signals=signals,
        prices=prices
    )
"""
