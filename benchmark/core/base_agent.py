from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Callable
import os
from pathlib import Path
from datetime import datetime

class BaseAgent(ABC):
    """
    Abstract base class for all task agents.
    Each agent must implement this interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.hyperparameters = {}
        self.is_setup = False
        
        # Tools management
        self.available_tools = {}
        self.enabled_tools = []
        
        # Simple token tracking
        self.run_tokens = []
        
    def setup(self) -> None:
        """
        Basic setup function that can be overridden by subclasses.
        Performs common initialization tasks.
        """
        self.agent_name = self.config.get("name", self.__class__.__name__)
        self.agent_description = self.config.get("description", "No description provided.")
        
        print(f"Setting up agent: {self.agent_name}")
        print(f"Description: {self.agent_description}")
        
        # Initialize default hyperparameters if provided in config
        default_hyperparams = self.config.get("hyperparameters", {})
        self.hyperparameters.update(default_hyperparams)
        
        # Initialize tools if provided in config
        tools_config = self.config.get("tools", {})
        if tools_config:
            self.configure_tools(tools_config)
        
        self.is_setup = True
        print("Agent setup completed.")
    
    @abstractmethod
    def run_task(self, prompt: str, path: str, tools: Optional[List[str]] = None) -> Any:
        """
        Run the agent with the given prompt in the specified directory.
        
        Args:
            prompt: The task prompt/instruction
            path: The working directory path for the task
            tools: Optional list of specific tools to use for this task
            
        Returns:
            Any: The result of running the task (format depends on specific agent)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Returns information about the agent including name, description, 
        capabilities, and current hyperparameters.
        
        Returns:
            Dict containing agent information
        """
        pass
    
    @abstractmethod  
    def set_hyperparameters(self, **kwargs) -> None:
        """
        Set or update hyperparameters for the agent.
        
        Args:
            **kwargs: Hyperparameter key-value pairs
        """
        pass
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get current hyperparameters.
        
        Returns:
            Dict of current hyperparameter values
        """
        return self.hyperparameters.copy()
    
    # Tools management methods
    def register_tool(self, name: str, tool_func: Callable, description: str = "") -> None:
        """
        Register a tool that the agent can use.
        
        Args:
            name: Tool name/identifier
            tool_func: The callable tool function
            description: Optional description of what the tool does
        """
        self.available_tools[name] = {
            "function": tool_func,
            "description": description,
            "registered_at": datetime.now().isoformat()
        }
        print(f"Registered tool: {name}")
    
    def enable_tools(self, tool_names: List[str]) -> None:
        """
        Enable specific tools for use by the agent.
        
        Args:
            tool_names: List of tool names to enable
            
        Raises:
            ValueError: If any tool name is not registered
        """
        for tool_name in tool_names:
            if tool_name not in self.available_tools:
                raise ValueError(f"Tool '{tool_name}' is not registered")
        
        self.enabled_tools = tool_names.copy()
        print(f"Enabled tools: {', '.join(tool_names)}")
    
    def disable_tools(self, tool_names: Optional[List[str]] = None) -> None:
        """
        Disable specific tools or all tools.
        
        Args:
            tool_names: List of tool names to disable. If None, disables all tools.
        """
        if tool_names is None:
            self.enabled_tools = []
            print("Disabled all tools")
        else:
            for tool_name in tool_names:
                if tool_name in self.enabled_tools:
                    self.enabled_tools.remove(tool_name)
            print(f"Disabled tools: {', '.join(tool_names)}")
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered tools.
        
        Returns:
            Dict mapping tool names to their information
        """
        return {name: {"description": info["description"], 
                      "registered_at": info["registered_at"]}
                for name, info in self.available_tools.items()}
    
    def get_enabled_tools(self) -> List[str]:
        """
        Get list of currently enabled tools.
        
        Returns:
            List of enabled tool names
        """
        return self.enabled_tools.copy()
    
    def configure_tools(self, tools_config: Dict[str, Any]) -> None:
        """
        Configure tools from a configuration dictionary.
        
        Args:
            tools_config: Dictionary with tool configuration
        """
        # This is a placeholder that can be overridden by subclasses
        # for more sophisticated tool configuration
        enabled = tools_config.get("enabled", [])
        if enabled:
            # Assume tools are already registered or will be registered by subclass
            self.enabled_tools = enabled
    
    # Simple token tracking methods
    def log_token(self, token_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a token/step during task execution.
        
        Args:
            token_type: Type of token (e.g., "thought", "tool_call", "output", "decision")
            content: The token content
            metadata: Optional additional metadata
        """
        token = {
            "timestamp": datetime.now().isoformat(),
            "token_type": token_type,
            "content": content,
            "metadata": metadata or {}
        }
        self.run_tokens.append(token)
    
    def get_run_tokens(self, token_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all tokens from the current run, optionally filtered by type.
        
        Args:
            token_type: Optional token type to filter by
            
        Returns:
            List of token entries
        """
        if token_type is None:
            return self.run_tokens.copy()
        
        return [t for t in self.run_tokens if t.get("token_type") == token_type]
    
    def clear_run_tokens(self) -> None:
        """
        Clear all logged tokens from the current run.
        """
        self.run_tokens = []
      
    def ensure_setup(self) -> None:
        """
        Ensure the agent has been set up before running tasks.
        
        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if not self.is_setup:
            raise RuntimeError("Agent must be set up before running tasks. Call setup() first.")
    
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        Can be overridden by subclasses for more specific reset behavior.
        """
        self.hyperparameters = {}
        self.enabled_tools = []
        self.run_tokens = []
        self.is_setup = False
        print(f"Agent {self.agent_name} has been reset.")
