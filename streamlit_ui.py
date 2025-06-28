#!/usr/bin/env python3
"""Streamlit UI for Custom LLM Chatbot."""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Custom LLM Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #1f77b4;
}

.user-message {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}

.assistant-message {
    background-color: #f3e5f5;
    border-left-color: #9c27b0;
}

.metrics-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online {
    background-color: #4caf50;
}

.status-offline {
    background-color: #f44336;
}
</style>
""", unsafe_allow_html=True)

class LLMClient:
    """Client for communicating with the LLM API server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_health(self) -> Dict[str, Any]:
        """Check server health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Stats request failed: {e}")
            return {"error": str(e)}
    
    def generate_text(self, 
                     prompt: str,
                     max_tokens: int = 512,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     repetition_penalty: float = 1.1,
                     stop_sequences: Optional[List[str]] = None,
                     stream: bool = False) -> Dict[str, Any]:
        """Generate text using the LLM."""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop_sequences": stop_sequences,
                "stream": stream
            }
            
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Generation request failed: {e}")
            return {"error": str(e)}

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = LLMClient()
    if "server_status" not in st.session_state:
        st.session_state.server_status = None
    if "stats" not in st.session_state:
        st.session_state.stats = None

def display_server_status():
    """Display server status in the sidebar."""
    st.sidebar.markdown("### 🖥️ Server Status")
    
    # Check server health
    health = st.session_state.client.check_health()
    
    if "error" in health:
        st.sidebar.markdown(
            '<div class="status-indicator status-offline"></div>**Offline**',
            unsafe_allow_html=True
        )
        st.sidebar.error(f"Connection Error: {health['error']}")
        return False
    
    # Display status
    if health.get("status") == "healthy":
        st.sidebar.markdown(
            '<div class="status-indicator status-online"></div>**Online**',
            unsafe_allow_html=True
        )
        
        # Display detailed status
        st.sidebar.success("✅ Server is healthy")
        
        if health.get("is_loaded"):
            st.sidebar.success("✅ Model loaded")
        else:
            st.sidebar.warning("⚠️ Model not loaded")
            
        if health.get("is_ready"):
            st.sidebar.success("✅ Ready for inference")
        else:
            st.sidebar.warning("⚠️ Not ready for inference")
            
        # Display backend info
        backend = health.get("backend_type", "Unknown")
        st.sidebar.info(f"Backend: {backend}")
        
        # GPU info
        if health.get("gpu_available"):
            gpu_count = health.get("gpu_count", 0)
            st.sidebar.info(f"GPUs: {gpu_count} available")
        
        return True
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-offline"></div>**Offline**',
            unsafe_allow_html=True
        )
        st.sidebar.error("Server not healthy")
        return False

def display_server_stats():
    """Display server statistics."""
    stats = st.session_state.client.get_stats()
    
    if "error" not in stats:
        st.sidebar.markdown("### 📊 Server Statistics")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Requests", stats.get("request_count", 0))
            st.metric("Avg Time/Req", f"{stats.get('avg_time_per_request', 0):.2f}s")
        
        with col2:
            st.metric("Total Tokens", stats.get("total_tokens_generated", 0))
            st.metric("Tokens/Sec", f"{stats.get('avg_tokens_per_second', 0):.1f}")

def display_generation_settings():
    """Display generation parameter controls."""
    st.sidebar.markdown("### ⚙️ Generation Settings")
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=1,
        max_value=2048,
        value=512,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in generation"
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling parameter"
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        help="Top-k sampling parameter"
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.1,
        help="Penalty for repeating tokens"
    )
    
    # Stop sequences
    stop_sequences_text = st.sidebar.text_area(
        "Stop Sequences (one per line)",
        value="",
        help="Sequences that will stop generation"
    )
    
    stop_sequences = None
    if stop_sequences_text.strip():
        stop_sequences = [seq.strip() for seq in stop_sequences_text.split("\n") if seq.strip()]
    
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "stop_sequences": stop_sequences
    }

def display_chat_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{content}</div>',
            unsafe_allow_html=True
        )
        
        # Display metadata if available
        if metadata:
            with st.expander("📊 Generation Details"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tokens Generated", metadata.get("tokens_generated", 0))
                    st.metric("Generation Time", f"{metadata.get('generation_time', 0):.2f}s")
                
                with col2:
                    st.metric("Tokens/Second", f"{metadata.get('tokens_per_second', 0):.1f}")
                    st.metric("Finish Reason", metadata.get("finish_reason", "unknown"))
                
                with col3:
                    if "request_id" in metadata:
                        st.text(f"Request ID: {metadata['request_id']}")

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Custom LLM Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Control Panel")
        
        # Server status
        server_online = display_server_status()
        
        if server_online:
            # Server statistics
            display_server_stats()
            
            # Generation settings
            generation_params = display_generation_settings()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        else:
            st.error("Please ensure the LLM server is running on http://localhost:8000")
            st.markdown("""
            **To start the server:**
            ```bash
            python serve.py --server-type vllm --model-path your-model-path
            ```
            """)
    
    # Main chat interface
    if server_online:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("metadata")
            )
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)
            
            # Generate response
            with st.spinner("Generating response..."):
                start_time = time.time()
                
                response = st.session_state.client.generate_text(
                    prompt=prompt,
                    **generation_params
                )
                
                end_time = time.time()
                
                if "error" in response:
                    st.error(f"Generation failed: {response['error']}")
                else:
                    # Extract response text and metadata
                    response_text = response.get("text", "")
                    metadata = {
                        "tokens_generated": response.get("tokens_generated", 0),
                        "generation_time": response.get("generation_time", end_time - start_time),
                        "tokens_per_second": response.get("tokens_per_second", 0),
                        "finish_reason": response.get("finish_reason", "completed"),
                        "request_id": response.get("request_id", "")
                    }
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "metadata": metadata
                    })
                    
                    # Display the response
                    display_chat_message("assistant", response_text, metadata)
                    
                    # Auto-scroll to bottom
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Custom LLM Chatbot - Powered by vLLM and Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()