#!/usr/bin/env python3
"""
Terra & Luna Analytics - Main Application Entry Point
NASA Space Apps Challenge 2025
Óbuda University Space Lab
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from analytics.core import AnalyticsEngine
from visualization.dashboard import create_dashboard


def main():
    """Main application entry point."""
    print("🚀 Terra & Luna Analytics - NASA Space Apps 2025")
    print("=" * 50)
    
    # Initialize analytics engine
    engine = AnalyticsEngine()
    print("✅ Analytics engine initialized")
    
    # Create and launch dashboard
    print("🌐 Starting visualization dashboard...")
    app = create_dashboard(engine)
    
    # Run the dashboard
    if __name__ == "__main__":
        app.run_server(debug=True, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    main()