"""Abort controller for graceful agent interruption."""

import sys
import os
import threading
import time
import select
from typing import Optional, Callable

# Platform-specific imports for raw keyboard input
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False  # Windows doesn't have termios


class AgentAbortedError(Exception):
    """Exception raised when agent execution is aborted by user."""
    pass


class AbortController:
    """Controls abortion of agent execution.
    
    Usage:
        controller = AbortController()
        
        # In background thread or signal handler:
        controller.abort()
        
        # In agent loop:
        controller.check()  # Raises AgentAbortedError if aborted
    """
    
    def __init__(self):
        self._aborted = False
        self._lock = threading.Lock()
    
    @property
    def is_aborted(self) -> bool:
        """Check if abort was requested."""
        with self._lock:
            return self._aborted
    
    def abort(self) -> None:
        """Request abortion of current operation."""
        with self._lock:
            self._aborted = True
    
    def reset(self) -> None:
        """Reset abort state for new operation."""
        with self._lock:
            self._aborted = False
    
    def check(self) -> None:
        """Check if aborted and raise exception if so.
        
        Raises:
            AgentAbortedError: If abort was requested
        """
        if self.is_aborted:
            raise AgentAbortedError("Agent execution aborted by user")


class InterruptHandler:
    """Handles double-ESC interrupt pattern.
    
    Tracks ESC key presses with a timeout window. When two presses
    occur within the timeout, triggers the abort callback.
    """
    
    def __init__(
        self, 
        on_interrupt: Callable[[], None],
        timeout: float = 0.5,
        on_first_press: Optional[Callable[[], None]] = None
    ):
        """Initialize the interrupt handler.
        
        Args:
            on_interrupt: Callback when double-ESC triggers interrupt
            timeout: Time window for double-press (seconds)
            on_first_press: Optional callback after first ESC press
        """
        self.on_interrupt = on_interrupt
        self.on_first_press = on_first_press
        self.timeout = timeout
        self.esc_count = 0
        self.last_esc_time = 0.0
        self._lock = threading.Lock()
    
    def handle_esc(self) -> bool:
        """Handle an ESC key press.
        
        Returns:
            True if interrupt was triggered, False otherwise
        """
        with self._lock:
            now = time.time()
            
            # Reset counter if timeout expired
            if now - self.last_esc_time > self.timeout:
                self.esc_count = 0
            
            self.esc_count += 1
            self.last_esc_time = now
            
            if self.esc_count >= 2:
                # Double-ESC detected - trigger interrupt
                self.on_interrupt()
                self.esc_count = 0
                return True
            else:
                # First ESC - notify user
                if self.on_first_press:
                    self.on_first_press()
                return False
    
    def reset(self) -> None:
        """Reset the ESC counter."""
        with self._lock:
            self.esc_count = 0
            self.last_esc_time = 0.0


class KeyboardListener:
    """Background thread for listening to keyboard input.
    
    Runs in a separate thread and monitors stdin for ESC key presses.
    When ESC is detected, forwards to the interrupt handler.
    
    Only works on Unix-like systems with termios support.
    """
    
    ESC_CHAR = '\x1b'  # ESC character
    
    def __init__(self, interrupt_handler: InterruptHandler):
        """Initialize keyboard listener.
        
        Args:
            interrupt_handler: Handler for ESC key presses
        """
        self.interrupt_handler = interrupt_handler
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._old_settings = None
        self._active = False
    
    @property
    def is_available(self) -> bool:
        """Check if keyboard listening is available on this platform."""
        return HAS_TERMIOS and sys.stdin.isatty()
    
    def start(self) -> bool:
        """Start listening for keyboard input.
        
        Returns:
            True if listener started successfully, False otherwise
        """
        if not self.is_available:
            return False
        
        if self._active:
            return True  # Already running
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        self._active = True
        return True
    
    def stop(self) -> None:
        """Stop listening for keyboard input."""
        if not self._active:
            return
        
        self._stop_event.set()
        self._active = False
        
        # Thread will exit on next iteration or timeout
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
    
    def _listen_loop(self) -> None:
        """Main listening loop (runs in background thread)."""
        if not HAS_TERMIOS:
            return
        
        fd = sys.stdin.fileno()
        
        try:
            # Save current terminal settings
            self._old_settings = termios.tcgetattr(fd)
            
            # Set terminal to raw mode (no buffering, no echo)
            tty.setraw(fd)
            
            while not self._stop_event.is_set():
                # Use select for non-blocking read with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    
                    if char == self.ESC_CHAR:
                        # ESC detected - handle it
                        self.interrupt_handler.handle_esc()
                        
        except Exception as e:
            print(f"DEBUG: KeyboardListener error: {e}")
            pass  # Silently ignore errors in background thread
            
        finally:
            # Restore terminal settings
            if self._old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
                except Exception:
                    pass
