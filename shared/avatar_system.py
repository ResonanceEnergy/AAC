#!/usr/bin/env python3
"""
AAC Avatar System
=================

Real-time avatar animation system synchronized with text-to-speech.
Provides visual mouth movements and facial expressions for AZ responses.
"""

import cv2
import numpy as np
import threading
import time
import queue
from typing import Optional, Tuple, List
from pathlib import Path
import base64
import io
from PIL import Image
import streamlit as st

class AACAvatarSystem:
    """
    Avatar animation system with lip-sync capabilities.
    Provides real-time mouth movements synchronized with speech.
    """

    def __init__(self):
        self.is_speaking = False
        self.current_phoneme = "rest"
        self.animation_queue = queue.Queue()
        self.avatar_image = None
        self.mouth_shapes = self._define_mouth_shapes()
        self._load_base_avatar()

    def _load_base_avatar(self):
        """Load or generate base avatar image"""
        try:
            # Create a simple avatar programmatically
            self.avatar_image = self._create_base_avatar()
        except Exception as e:
            print(f"Warning: Could not create avatar: {e}")
            # Fallback to a simple colored rectangle
            self.avatar_image = np.zeros((200, 200, 3), dtype=np.uint8)
            self.avatar_image[:] = [100, 150, 200]  # Blue background

    def _create_base_avatar(self) -> np.ndarray:
        """Create a simple cartoon avatar"""
        # Create 200x200 RGB image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = [240, 248, 255]  # Light blue background

        # Draw face (circle)
        center = (100, 100)
        cv2.circle(img, center, 80, (255, 218, 185), -1)  # Peach face

        # Draw eyes
        cv2.circle(img, (80, 85), 8, (255, 255, 255), -1)  # Left eye white
        cv2.circle(img, (120, 85), 8, (255, 255, 255), -1)  # Right eye white
        cv2.circle(img, (80, 85), 4, (0, 0, 0), -1)  # Left pupil
        cv2.circle(img, (120, 85), 4, (0, 0, 0), -1)  # Right pupil

        # Draw nose (small circle)
        cv2.circle(img, (100, 105), 3, (255, 200, 150), -1)

        # Draw mouth area (will be animated)
        cv2.ellipse(img, (100, 125), (15, 8), 0, 0, 360, (200, 100, 100), -1)

        return img

    def _define_mouth_shapes(self) -> dict:
        """Define different mouth shapes for phonemes"""
        return {
            "rest": {"width": 15, "height": 8, "color": (200, 100, 100)},
            "AH": {"width": 20, "height": 12, "color": (150, 50, 50)},
            "EE": {"width": 25, "height": 6, "color": (200, 100, 100)},
            "OH": {"width": 18, "height": 15, "color": (150, 50, 50)},
            "OO": {"width": 12, "height": 12, "color": (180, 80, 80)},
            "MM": {"width": 8, "height": 4, "color": (220, 120, 120)},
            "SS": {"width": 22, "height": 3, "color": (200, 100, 100)},
            "FF": {"width": 16, "height": 5, "color": (200, 100, 100)},
        }

    def _animate_mouth(self, phoneme: str) -> np.ndarray:
        """Animate mouth based on phoneme"""
        if self.avatar_image is None:
            return np.zeros((200, 200, 3), dtype=np.uint8)

        # Copy base image
        animated = self.avatar_image.copy()

        # Get mouth shape
        shape = self.mouth_shapes.get(phoneme, self.mouth_shapes["rest"])

        # Draw animated mouth
        center = (100, 125)
        cv2.ellipse(animated, center, (shape["width"], shape["height"]),
                   0, 0, 360, shape["color"], -1)

        return animated

    def _phoneme_from_text(self, text: str) -> List[Tuple[str, float]]:
        """Simple phoneme extraction from text (basic approximation)"""
        phonemes = []
        text = text.lower()

        # Simple vowel/consonant mapping
        vowel_map = {
            'a': 'AH', 'e': 'EE', 'i': 'EE', 'o': 'OH', 'u': 'OO',
            'ah': 'AH', 'eh': 'EE', 'ih': 'EE', 'oh': 'OH', 'uh': 'OO'
        }

        consonant_map = {
            'm': 'MM', 'b': 'MM', 'p': 'MM',
            's': 'SS', 'z': 'SS', 'f': 'FF', 'v': 'FF'
        }

        for char in text:
            if char in vowel_map:
                phonemes.append((vowel_map[char], 0.1))
            elif char in consonant_map:
                phonemes.append((consonant_map[char], 0.08))
            elif char == ' ':
                phonemes.append(('rest', 0.05))
            else:
                phonemes.append(('rest', 0.03))

        return phonemes

    def speak_with_animation(self, text: str, duration: float = None):
        """Speak text with synchronized mouth animation"""
        if not text:
            return

        self.is_speaking = True

        # Extract phonemes
        phoneme_sequence = self._phoneme_from_text(text)

        # Calculate duration if not provided
        if duration is None:
            duration = len(text) * 0.08  # Rough estimate

        # Animate through phonemes
        start_time = time.time()
        phoneme_idx = 0

        while phoneme_idx < len(phoneme_sequence) and self.is_speaking:
            phoneme, phoneme_duration = phoneme_sequence[phoneme_idx]

            # Update current phoneme
            self.current_phoneme = phoneme

            # Wait for phoneme duration
            time.sleep(min(phoneme_duration, 0.1))

            phoneme_idx += 1

            # Check if we've exceeded total duration
            if time.time() - start_time > duration:
                break

        # Return to rest position
        self.current_phoneme = "rest"
        self.is_speaking = False

    def get_current_frame(self) -> np.ndarray:
        """Get current animated frame"""
        return self._animate_mouth(self.current_phoneme)

    def get_frame_as_base64(self) -> str:
        """Get current frame as base64 string for Streamlit"""
        frame = self.get_current_frame()
        if frame is None:
            return ""

        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    def start_speaking_animation(self, text: str):
        """Start speaking animation in background thread"""
        def animate():
            self.speak_with_animation(text)

        thread = threading.Thread(target=animate, daemon=True)
        thread.start()

    def stop_speaking(self):
        """Stop current speech animation"""
        self.is_speaking = False
        self.current_phoneme = "rest"


class AACAvatarManager:
    """Manager for avatar instances and coordination"""

    def __init__(self):
        self.avatars = {}
        self.active_avatar = None

    def get_avatar(self, name: str = "az") -> AACAvatarSystem:
        """Get or create avatar instance"""
        if name not in self.avatars:
            self.avatars[name] = AACAvatarSystem()

        self.active_avatar = self.avatars[name]
        return self.avatars[name]

    def speak_text(self, text: str, avatar_name: str = "az"):
        """Make specified avatar speak text"""
        avatar = self.get_avatar(avatar_name)
        avatar.start_speaking_animation(text)


# Global manager instance
_avatar_manager = None

def get_avatar_manager() -> AACAvatarManager:
    """Get global avatar manager"""
    global _avatar_manager
    if _avatar_manager is None:
        _avatar_manager = AACAvatarManager()
    return _avatar_manager


def create_streamlit_avatar_component():
    """Create Streamlit component for avatar display"""
    avatar_manager = get_avatar_manager()
    avatar = avatar_manager.get_avatar("az")

    # Create placeholder for avatar
    avatar_placeholder = st.empty()

    # Display initial avatar
    avatar_placeholder.image(avatar.get_frame_as_base64(), caption="AZ Avatar", width=200)

    return avatar_placeholder, avatar


if __name__ == "__main__":
    # Test avatar system
    manager = get_avatar_manager()
    avatar = manager.get_avatar("az")

    # Test animation
    test_text = "Hello, I am AZ, your AI assistant for the AAC system."
    print(f"Animating text: {test_text}")

    avatar.start_speaking_animation(test_text)

    # Show frames for a few seconds
    for i in range(50):
        frame = avatar.get_current_frame()
        print(f"Frame {i}: phoneme={avatar.current_phoneme}, speaking={avatar.is_speaking}")
        time.sleep(0.1)

    print("Animation test complete")