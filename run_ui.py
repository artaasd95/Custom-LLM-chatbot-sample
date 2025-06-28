#!/usr/bin/env python3
"""Startup script for the Streamlit UI."""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Streamlit UI for Custom LLM Chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the Streamlit server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to bind the Streamlit server to"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="ui_config.yaml",
        help="Path to UI configuration file"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the LLM API server"
    )
    
    parser.add_argument(
        "--app-file",
        type=str,
        default="streamlit_app.py",
        help="Streamlit app file to run"
    )
    
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="light",
        help="UI theme"
    )
    
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Automatically open browser"
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import requests
        import yaml
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

def check_server_connection(server_url: str) -> bool:
    """Check if the LLM server is accessible."""
    try:
        import requests
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def update_config_file(config_path: str, server_url: str):
    """Update configuration file with server URL."""
    try:
        import yaml
        
        # Load existing config or create new one
        config = {}
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        
        # Update server URL
        if 'server' not in config:
            config['server'] = {}
        config['server']['base_url'] = server_url
        
        # Save updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Updated configuration file: {config_path}")
        
    except Exception as e:
        print(f"Warning: Could not update config file: {e}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    print("✅ Dependencies OK")
    
    # Check if app file exists
    app_path = Path(args.app_file)
    if not app_path.exists():
        print(f"Error: Streamlit app file not found: {args.app_file}")
        sys.exit(1)
    
    # Update config file with server URL
    update_config_file(args.config, args.server_url)
    
    # Check server connection
    print(f"Checking connection to LLM server at {args.server_url}...")
    if check_server_connection(args.server_url):
        print("✅ LLM server is accessible")
    else:
        print(f"⚠️  Warning: LLM server at {args.server_url} is not accessible")
        print("Make sure the server is running with:")
        print("python serve.py --server-type vllm --model-path your-model-path")
        print("")
        print("Continuing anyway - you can start the server later...")
    
    # Prepare Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--theme.base", args.theme
    ]
    
    if not args.browser:
        cmd.extend(["--server.headless", "true"])
    
    # Set environment variables
    env = os.environ.copy()
    env["STREAMLIT_CONFIG_FILE"] = args.config
    
    print(f"\n🚀 Starting Streamlit UI...")
    print(f"📱 UI will be available at: http://{args.host}:{args.port}")
    print(f"🔗 LLM Server: {args.server_url}")
    print(f"⚙️  Config file: {args.config}")
    print(f"🎨 Theme: {args.theme}")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit UI...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()