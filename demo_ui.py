#!/usr/bin/env python3
"""Quick demo script to test the Streamlit UI with a mock server."""

import os
import sys
import time
import subprocess
import threading
import argparse
from pathlib import Path

def run_demo_server(host="localhost", port=8000):
    """Run the demo server in a separate process."""
    try:
        cmd = [sys.executable, "demo_server.py", "--host", host, "--port", str(port)]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Demo server error: {e}")

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import streamlit
    except ImportError:
        missing.append("streamlit")
    
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def wait_for_server(url, timeout=30):
    """Wait for server to be ready."""
    import requests
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    
    return False

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo the Streamlit UI with mock server")
    parser.add_argument("--server-host", default="localhost", help="Demo server host")
    parser.add_argument("--server-port", type=int, default=8000, help="Demo server port")
    parser.add_argument("--ui-host", default="localhost", help="UI host")
    parser.add_argument("--ui-port", type=int, default=8501, help="UI port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    
    print("🚀 Custom LLM Chatbot - Demo Mode")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check if files exist
    required_files = ["demo_server.py", "streamlit_app.py", "ui_config.yaml"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return 1
    
    server_url = f"http://{args.server_host}:{args.server_port}"
    ui_url = f"http://{args.ui_host}:{args.ui_port}"
    
    print(f"📡 Demo Server: {server_url}")
    print(f"🖥️  UI: {ui_url}")
    print("")
    
    try:
        # Start demo server in background thread
        print("Starting demo server...")
        server_thread = threading.Thread(
            target=run_demo_server,
            args=(args.server_host, args.server_port),
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to be ready
        print("Waiting for server to start...")
        if not wait_for_server(server_url):
            print("❌ Demo server failed to start")
            return 1
        
        print("✅ Demo server is ready")
        
        # Start Streamlit UI
        print("Starting Streamlit UI...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.address", args.ui_host,
            "--server.port", str(args.ui_port),
            "--theme.base", "light"
        ]
        
        if args.no_browser:
            cmd.extend(["--server.headless", "true"])
        
        # Set environment to point to demo server
        env = os.environ.copy()
        
        print("")
        print("🎉 Demo is ready!")
        print(f"🌐 Open your browser to: {ui_url}")
        print(f"🤖 The UI will connect to the demo server at: {server_url}")
        print("")
        print("💡 Try these demo prompts:")
        print("   - 'Write a Python function to sort a list'")
        print("   - 'Tell me a creative story about AI'")
        print("   - 'What is machine learning?'")
        print("   - 'Hello, how are you?'")
        print("")
        print("Press Ctrl+C to stop the demo")
        print("-" * 40)
        
        # Update config to point to demo server
        try:
            import yaml
            config_path = "ui_config.yaml"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['server']['base_url'] = server_url
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Warning: Could not update config: {e}")
        
        # Run Streamlit
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())