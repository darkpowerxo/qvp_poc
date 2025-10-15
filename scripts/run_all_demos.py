"""
Comprehensive Demo Script

Demonstrates all QVP platform capabilities:
- Docker containerization
- Interactive dashboards
- Live trading simulation
- WebSocket data feeds
- Real-time risk monitoring
"""

import asyncio
import subprocess
import time
from pathlib import Path
from loguru import logger


def run_command(cmd: str, description: str):
    """Run a shell command and display status."""
    logger.info(f"▶ {description}")
    logger.info(f"  Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.success(f"✓ {description} - SUCCESS")
            if result.stdout:
                print(result.stdout)
        else:
            logger.error(f"✗ {description} - FAILED")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"✗ {description} - ERROR: {e}")
        return False


def demo_traditional_backtest():
    """Run traditional backtesting demo."""
    logger.info("="*80)
    logger.info("DEMO 1: TRADITIONAL BACKTESTING")
    logger.info("="*80)
    
    run_command(
        "uv run python scripts/run_demo.py",
        "Running traditional backtest demo"
    )
    
    logger.info("\n✓ Results saved to data/results/")
    time.sleep(2)


async def demo_live_simulation():
    """Run live trading simulation."""
    logger.info("="*80)
    logger.info("DEMO 2: LIVE TRADING SIMULATION")
    logger.info("="*80)
    
    from qvp.live.simulator import LiveSimulator
    from qvp.live.async_strategy import SimpleVIXMeanReversion
    
    # Create simulator
    sim = LiveSimulator(
        initial_capital=1_000_000,
        symbols=["SPY", "^VIX"],
        initial_prices={"SPY": 450.0, "^VIX": 18.0}
    )
    
    # Add strategy
    strategy = SimpleVIXMeanReversion(
        portfolio=sim.portfolio,
        vix_threshold_high=22.0,
        vix_threshold_low=16.0,
        position_size=100_000
    )
    sim.add_strategy(strategy)
    
    # Run for 30 seconds
    logger.info("Running live simulation for 30 seconds...")
    await sim.start(duration=30)
    
    logger.info("\n✓ Results saved to data/live_sim/")
    time.sleep(2)


def demo_dashboard():
    """Launch interactive dashboard."""
    logger.info("="*80)
    logger.info("DEMO 3: INTERACTIVE DASHBOARD")
    logger.info("="*80)
    
    logger.info("Starting dashboard on http://localhost:8050")
    logger.info("Press Ctrl+C to stop the dashboard")
    
    run_command(
        "uv run python -m qvp.dashboard.app",
        "Launching main dashboard"
    )


def demo_risk_monitor():
    """Launch risk monitoring dashboard."""
    logger.info("="*80)
    logger.info("DEMO 4: RISK MONITORING DASHBOARD")
    logger.info("="*80)
    
    logger.info("Starting risk monitor on http://localhost:8052")
    logger.info("Press Ctrl+C to stop the dashboard")
    
    run_command(
        "uv run python -m qvp.dashboard.risk_monitor",
        "Launching risk monitoring dashboard"
    )


def demo_docker():
    """Demonstrate Docker deployment."""
    logger.info("="*80)
    logger.info("DEMO 5: DOCKER CONTAINERIZATION")
    logger.info("="*80)
    
    logger.info("Docker commands available:")
    logger.info("  Build image:     docker build -t qvp-platform .")
    logger.info("  Run container:   docker run --rm qvp-platform")
    logger.info("  Start services:  docker-compose up -d")
    logger.info("  View logs:       docker-compose logs -f")
    logger.info("  Stop services:   docker-compose down")
    logger.info("")
    logger.info("Helper scripts:")
    logger.info("  PowerShell:  . .\\scripts\\docker-helpers.ps1")
    logger.info("  Bash:        source ./scripts/docker.sh")


def show_menu():
    """Display demo menu."""
    print("\n" + "="*80)
    print(" " * 20 + "QVP PLATFORM - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nAvailable Demos:")
    print("  1. Traditional Backtesting (Historical Data)")
    print("  2. Live Trading Simulation (Async + WebSocket Feeds)")
    print("  3. Interactive Dashboard (Plotly/Dash)")
    print("  4. Risk Monitoring Dashboard (Real-time)")
    print("  5. Docker Containerization (Deployment)")
    print("  6. Run All Demos Sequentially")
    print("  0. Exit")
    print("="*80)


async def main():
    """Main demo orchestrator."""
    logger.info("QVP Platform - Comprehensive Demo")
    
    while True:
        show_menu()
        choice = input("\nSelect demo (0-6): ").strip()
        
        if choice == "0":
            logger.info("Exiting demo. Thank you!")
            break
        
        elif choice == "1":
            demo_traditional_backtest()
        
        elif choice == "2":
            await demo_live_simulation()
        
        elif choice == "3":
            demo_dashboard()
        
        elif choice == "4":
            demo_risk_monitor()
        
        elif choice == "5":
            demo_docker()
        
        elif choice == "6":
            logger.info("Running all demos sequentially...")
            demo_traditional_backtest()
            await demo_live_simulation()
            logger.info("\n✓ All automated demos complete!")
            logger.info("\nInteractive demos (Dashboard & Risk Monitor) must be run separately.")
            logger.info("Use options 3 or 4 to launch them.")
        
        else:
            logger.warning("Invalid choice. Please select 0-6.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
