#!/usr/bin/env python3
"""Enhanced Streamlit UI for Custom LLM Chatbot with configuration support."""

import streamlit as st
import requests
import json
import time
import yaml
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the Streamlit UI."""
    
    def __init__(self, config_path: str = "ui_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "server": {
                "base_url": "http://localhost:8000",
                "timeout": 60
            },
            "ui": {
                "title": "Custom LLM Chatbot",
                "page_icon": "🤖",
                "layout": "wide"
            },
            "generation": {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1
            }
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

class LLMClient:
    """Enhanced client for communicating with the LLM API server."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.base_url = self.config.get("server.base_url", "http://localhost:8000")
        self.timeout = self.config.get("server.timeout", 60)
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
    
    def generate_text(self, **kwargs) -> Dict[str, Any]:
        """Generate text using the LLM."""
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=kwargs,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Generation request failed: {e}")
            return {"error": str(e)}

def initialize_session_state(config_manager: ConfigManager):
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = LLMClient(config_manager)
    if "config" not in st.session_state:
        st.session_state.config = config_manager
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = "General Chat"

def setup_page_config(config_manager: ConfigManager):
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title=config_manager.get("ui.title", "Custom LLM Chatbot"),
        page_icon=config_manager.get("ui.page_icon", "🤖"),
        layout=config_manager.get("ui.layout", "wide"),
        initial_sidebar_state="expanded"
    )

def load_custom_css(config_manager: ConfigManager):
    """Load custom CSS styling."""
    primary_color = config_manager.get("styling.primary_color", "#1f77b4")
    secondary_color = config_manager.get("styling.secondary_color", "#9c27b0")
    user_bg = config_manager.get("styling.user_message_bg", "#e3f2fd")
    assistant_bg = config_manager.get("styling.assistant_message_bg", "#f3e5f5")
    online_color = config_manager.get("styling.online_color", "#4caf50")
    offline_color = config_manager.get("styling.offline_color", "#f44336")
    
    st.markdown(f"""
    <style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {primary_color};
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .chat-message {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid {primary_color};
    }}
    
    .user-message {{
        background-color: {user_bg};
        border-left-color: {primary_color};
    }}
    
    .assistant-message {{
        background-color: {assistant_bg};
        border-left-color: {secondary_color};
    }}
    
    .status-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }}
    
    .status-online {{
        background-color: {online_color};
    }}
    
    .status-offline {{
        background-color: {offline_color};
    }}
    
    .template-card {{
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        cursor: pointer;
        transition: all 0.2s;
    }}
    
    .template-card:hover {{
        background-color: #e9ecef;
        border-color: {primary_color};
    }}
    
    .template-selected {{
        background-color: {user_bg};
        border-color: {primary_color};
        border-width: 2px;
    }}
    </style>
    """, unsafe_allow_html=True)

def display_server_status(client: LLMClient) -> bool:
    """Display server status in the sidebar."""
    st.sidebar.markdown("### 🖥️ Server Status")
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    health = client.check_health()
    
    if "error" in health:
        st.sidebar.markdown(
            '<div class="status-indicator status-offline"></div>**Offline**',
            unsafe_allow_html=True
        )
        st.sidebar.error(f"Connection Error: {health['error']}")
        return False
    
    if health.get("status") == "healthy":
        st.sidebar.markdown(
            '<div class="status-indicator status-online"></div>**Online**',
            unsafe_allow_html=True
        )
        
        # Status indicators
        status_items = [
            ("Model loaded", health.get("is_loaded", False)),
            ("Ready for inference", health.get("is_ready", False)),
            ("GPU available", health.get("gpu_available", False))
        ]
        
        for label, status in status_items:
            if status:
                st.sidebar.success(f"✅ {label}")
            else:
                st.sidebar.warning(f"⚠️ {label}")
        
        # Additional info
        backend = health.get("backend_type", "Unknown")
        st.sidebar.info(f"Backend: {backend}")
        
        if health.get("gpu_available"):
            gpu_count = health.get("gpu_count", 0)
            st.sidebar.info(f"GPUs: {gpu_count} available")
        
        return True
    
    return False

def display_server_stats(client: LLMClient):
    """Display server statistics."""
    stats = client.get_stats()
    
    if "error" not in stats:
        st.sidebar.markdown("### 📊 Server Statistics")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Requests", stats.get("request_count", 0))
            st.metric("Avg Time/Req", f"{stats.get('avg_time_per_request', 0):.2f}s")
        
        with col2:
            st.metric("Total Tokens", stats.get("total_tokens_generated", 0))
            st.metric("Tokens/Sec", f"{stats.get('avg_tokens_per_second', 0):.1f}")

def display_prompt_templates(config_manager: ConfigManager):
    """Display prompt template selection."""
    templates = config_manager.get("prompt_templates", [])
    
    if templates:
        st.sidebar.markdown("### 📝 Prompt Templates")
        
        template_names = [t["name"] for t in templates]
        selected = st.sidebar.selectbox(
            "Choose a template:",
            template_names,
            index=template_names.index(st.session_state.selected_template) 
            if st.session_state.selected_template in template_names else 0
        )
        
        if selected != st.session_state.selected_template:
            st.session_state.selected_template = selected
        
        # Show template description
        selected_template = next((t for t in templates if t["name"] == selected), None)
        if selected_template:
            st.sidebar.info(selected_template["description"])

def display_generation_settings(config_manager: ConfigManager) -> Dict[str, Any]:
    """Display generation parameter controls."""
    st.sidebar.markdown("### ⚙️ Generation Settings")
    
    # Get default values and ranges from config
    defaults = config_manager.get("generation", {})
    ranges = config_manager.get("generation.ranges", {})
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=ranges.get("max_tokens", {}).get("min", 1),
        max_value=ranges.get("max_tokens", {}).get("max", 2048),
        value=defaults.get("max_tokens", 512),
        step=ranges.get("max_tokens", {}).get("step", 1),
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=ranges.get("temperature", {}).get("min", 0.0),
        max_value=ranges.get("temperature", {}).get("max", 2.0),
        value=defaults.get("temperature", 0.7),
        step=ranges.get("temperature", {}).get("step", 0.1),
        help="Controls randomness in generation"
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=ranges.get("top_p", {}).get("min", 0.0),
        max_value=ranges.get("top_p", {}).get("max", 1.0),
        value=defaults.get("top_p", 0.9),
        step=ranges.get("top_p", {}).get("step", 0.05),
        help="Nucleus sampling parameter"
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=ranges.get("top_k", {}).get("min", 1),
        max_value=ranges.get("top_k", {}).get("max", 100),
        value=defaults.get("top_k", 50),
        step=ranges.get("top_k", {}).get("step", 1),
        help="Top-k sampling parameter"
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=ranges.get("repetition_penalty", {}).get("min", 1.0),
        max_value=ranges.get("repetition_penalty", {}).get("max", 2.0),
        value=defaults.get("repetition_penalty", 1.1),
        step=ranges.get("repetition_penalty", {}).get("step", 0.1),
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

def apply_prompt_template(user_input: str, template_name: str, config_manager: ConfigManager) -> str:
    """Apply the selected prompt template to user input."""
    templates = config_manager.get("prompt_templates", [])
    selected_template = next((t for t in templates if t["name"] == template_name), None)
    
    if selected_template:
        return selected_template["template"].format(user_input=user_input)
    
    return user_input

def display_chat_message(role: str, content: str, metadata: Optional[Dict] = None, show_metadata: bool = True):
    """Display a chat message with proper styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message"><strong>You ({timestamp}):</strong><br>{content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message"><strong>Assistant ({timestamp}):</strong><br>{content}</div>',
            unsafe_allow_html=True
        )
        
        # Display metadata if available and enabled
        if metadata and show_metadata:
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

def export_conversation():
    """Export conversation history."""
    if st.session_state.messages:
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        
        json_str = json.dumps(conversation_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="💾 Export Conversation",
            data=json_str,
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

def main():
    """Main Streamlit application."""
    # Load configuration
    config_manager = ConfigManager()
    
    # Setup page
    setup_page_config(config_manager)
    load_custom_css(config_manager)
    
    # Initialize session state
    initialize_session_state(config_manager)
    
    # Header
    title = config_manager.get("ui.title", "Custom LLM Chatbot")
    st.markdown(f'<h1 class="main-header">{config_manager.get("ui.page_icon", "🤖")} {title}</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Control Panel")
        
        # Server status
        server_online = display_server_status(st.session_state.client)
        
        if server_online:
            # Server statistics
            if config_manager.get("ui.sidebar.show_statistics", True):
                display_server_stats(st.session_state.client)
            
            # Prompt templates
            if config_manager.get("features.enable_prompt_templates", True):
                display_prompt_templates(config_manager)
            
            # Generation settings
            if config_manager.get("ui.sidebar.show_generation_settings", True):
                generation_params = display_generation_settings(config_manager)
            else:
                generation_params = config_manager.get("generation", {})
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            
            with col2:
                if config_manager.get("features.enable_conversation_export", True):
                    export_conversation()
        
        else:
            st.error("Please ensure the LLM server is running")
            st.markdown(f"""
            **To start the server:**
            ```bash
            python serve.py --server-type vllm --model-path your-model-path --host {config_manager.get('server.host', 'localhost')} --port {config_manager.get('server.port', 8000)}
            ```
            """)
    
    # Main chat interface
    if server_online:
        # Display chat history
        show_metadata = config_manager.get("ui.chat.show_metadata", True)
        
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("metadata"),
                show_metadata
            )
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Apply prompt template
            processed_prompt = apply_prompt_template(
                prompt, 
                st.session_state.selected_template, 
                config_manager
            )
            
            # Add user message to history
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            display_chat_message("user", prompt)
            
            # Generate response
            with st.spinner("Generating response..."):
                start_time = time.time()
                
                response = st.session_state.client.generate_text(
                    prompt=processed_prompt,
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
                        "request_id": response.get("request_id", ""),
                        "template_used": st.session_state.selected_template
                    }
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Display the response
                    display_chat_message("assistant", response_text, metadata, show_metadata)
                    
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