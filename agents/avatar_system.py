"""
AAC Avatar System
=================

AI Avatar Interface with Speech Capabilities and Animation
Two avatars: AZ SUPREME (Strategic Advisor) and AX HELIX (Operations Commander)
"""

import asyncio
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import sys
from pathlib import Path
import os
import random
import re

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.communication_framework import get_communication_framework
from shared.super_agent_framework import get_super_agent_core

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AvatarEmotion(Enum):
    """Avatar emotional states"""
    FOCUSED = "focused"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    ANALYZING = "analyzing"
    DECISIVE = "decisive"
    CAUTIOUS = "cautious"
    OPTIMISTIC = "optimistic"
    ALERT = "alert"

class AvatarGesture(Enum):
    """Avatar gesture animations"""
    THINKING = "thinking"
    SPEAKING = "speaking"
    LISTENING = "listening"
    DECIDING = "deciding"
    ALERTING = "alerting"
    CELEBRATING = "celebrating"
    CONCERNED = "concerned"
    CONFIDENT = "confident"

@dataclass
class SpeechSynthesis:
    """Speech synthesis configuration"""
    voice: str = "en-US-Neural2-D"
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    emotion: str = "neutral"

@dataclass
class AvatarResponse:
    """Avatar response structure"""
    text: str
    emotion: AvatarEmotion
    gesture: AvatarGesture
    speech_config: SpeechSynthesis
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

class AvatarInterface:
    """
    AI Avatar Interface with Speech and Animation
    """

    def __init__(self, avatar_name: str, personality: str):
        self.avatar_name = avatar_name
        self.personality = personality
        self.logger = logging.getLogger(f"avatar.{avatar_name}")

        # Avatar state
        self.current_emotion = AvatarEmotion.FOCUSED
        self.current_gesture = AvatarGesture.LISTENING
        self.confidence_level = 0.95
        self.is_speaking = False
        self.is_listening = True

        # Speech synthesis
        self.speech_engine = self._initialize_speech_engine()

        # Animation state
        self.animation_queue = asyncio.Queue()
        self.gesture_duration = 2.0  # seconds

        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        self.response_templates = self._load_response_templates()

        # Knowledge base
        self.knowledge_base = self._load_knowledge_base()

    def _initialize_speech_engine(self) -> Any:
        """Initialize speech synthesis engine"""
        try:
            # Try to import speech libraries
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 0.9)
            return engine
        except ImportError:
            self.logger.warning("Speech synthesis not available, using text-only mode")
            return None

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load avatar response templates"""
        return {
            "greeting": [
                "Greetings. I am {avatar_name}, {personality}. How may I assist you?",
                "Hello. {avatar_name} here, ready to {role_action}.",
                "Good day. I am {avatar_name}, your {personality}. What can I do for you?"
            ],
            "analyzing": [
                "Analyzing the situation...",
                "Processing data and formulating response...",
                "Evaluating options and implications..."
            ],
            "decision": [
                "Based on my analysis, I recommend: {decision}",
                "My strategic assessment indicates: {decision}",
                "The optimal course of action is: {decision}"
            ],
            "concern": [
                "I detect potential issues that require attention.",
                "There are concerning developments that need monitoring.",
                "Risk indicators are elevated in this area."
            ],
            "confidence": [
                "I am confident in this assessment.",
                "The data supports this conclusion with high certainty.",
                "All indicators point to a positive outcome."
            ],
            "question": [
                "Could you provide more details on {topic}?",
                "I need additional information about {topic} to give a complete answer.",
                "Please clarify {topic} for better analysis."
            ]
        }

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load avatar knowledge base"""
        return {
            "financial_terms": {
                "var": "Value at Risk - measures potential loss in investment portfolio",
                "sharpe_ratio": "Risk-adjusted return measure comparing excess return to volatility",
                "drawdown": "Peak-to-trough decline in portfolio value",
                "alpha": "Excess return relative to market benchmark",
                "beta": "Measure of volatility relative to market"
            },
            "strategic_concepts": {
                "diversification": "Spreading investments to reduce risk",
                "hedging": "Protecting against adverse price movements",
                "arbitrage": "Exploiting price differences between markets",
                "momentum": "Continuing trend in asset prices",
                "mean_reversion": "Tendency of prices to return to historical average"
            },
            "personality_traits": {
                "supreme": {
                    "leadership": "Strategic oversight and supreme command",
                    "analysis": "Deep market and risk analysis",
                    "decision_making": "High-level strategic decisions",
                    "oversight": "System-wide monitoring and control"
                },
                "helix": {
                    "operations": "Operational excellence and efficiency",
                    "execution": "Tactical implementation and coordination",
                    "optimization": "Process and performance optimization",
                    "integration": "System integration and alignment"
                }
            }
        }

    async def process_query(self, query: str) -> AvatarResponse:
        """Process user query and generate response"""
        try:
            # Analyze query
            query_analysis = await self._analyze_query(query)

            # Generate response
            response = await self._generate_response(query_analysis)

            # Update avatar state
            await self._update_avatar_state(query_analysis, response)

            # Add to interaction history
            self.interaction_history.append({
                "query": query,
                "response": response.text,
                "emotion": response.emotion.value,
                "confidence": response.confidence,
                "timestamp": response.timestamp.isoformat()
            })

            # Trigger speech and animation
            await self._trigger_speech_and_animation(response)

            return response

        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            return AvatarResponse(
                text="I apologize, but I encountered an error processing your request. Please try again.",
                emotion=AvatarEmotion.CONCERNED,
                gesture=AvatarGesture.CONCERNED,
                speech_config=SpeechSynthesis(),
                confidence=0.5
            )

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query for intent and context"""
        query_lower = query.lower()

        analysis = {
            "intent": "unknown",
            "topics": [],
            "urgency": "normal",
            "complexity": "simple",
            "sentiment": "neutral",
            "entities": []
        }

        # Intent detection
        if any(word in query_lower for word in ["status", "how", "what", "report"]):
            analysis["intent"] = "inquiry"
        elif any(word in query_lower for word in ["analyze", "assess", "evaluate"]):
            analysis["intent"] = "analysis"
        elif any(word in query_lower for word in ["decide", "choose", "recommend"]):
            analysis["intent"] = "decision"
        elif any(word in query_lower for word in ["alert", "warning", "critical"]):
            analysis["intent"] = "alert"
            analysis["urgency"] = "high"

        # Topic extraction
        financial_terms = self.knowledge_base.get("financial_terms", {})
        for term in financial_terms.keys():
            if term in query_lower:
                analysis["topics"].append(term)

        # Urgency detection
        if any(word in query_lower for word in ["urgent", "critical", "immediate", "emergency"]):
            analysis["urgency"] = "critical"
        elif any(word in query_lower for word in ["important", "significant", "concerning"]):
            analysis["urgency"] = "high"

        # Complexity assessment
        if len(query.split()) > 20 or "?" in query:
            analysis["complexity"] = "complex"

        return analysis

    async def _generate_response(self, analysis: Dict[str, Any]) -> AvatarResponse:
        """Generate appropriate response based on query analysis"""
        intent = analysis.get("intent", "unknown")
        topics = analysis.get("topics", [])
        urgency = analysis.get("urgency", "normal")

        # Base response structure
        response_text = ""
        emotion = AvatarEmotion.FOCUSED
        gesture = AvatarGesture.SPEAKING
        speech_config = SpeechSynthesis()

        # Generate response based on intent
        if intent == "inquiry":
            response_text = await self._generate_inquiry_response(analysis)
        elif intent == "analysis":
            response_text = await self._generate_analysis_response(analysis)
            emotion = AvatarEmotion.ANALYZING
            gesture = AvatarGesture.THINKING
        elif intent == "decision":
            response_text = await self._generate_decision_response(analysis)
            emotion = AvatarEmotion.DECISIVE
            gesture = AvatarGesture.DECIDING
        elif intent == "alert":
            response_text = await self._generate_alert_response(analysis)
            emotion = AvatarEmotion.ALERT
            gesture = AvatarGesture.ALERTING
        else:
            response_text = self._get_random_template("greeting")

        # Adjust for urgency
        if urgency == "high":
            emotion = AvatarEmotion.CONCERNED
            speech_config.speed = 1.2
        elif urgency == "critical":
            emotion = AvatarEmotion.ALERT
            speech_config.speed = 1.3
            speech_config.volume = 1.1

        # Adjust speech for emotion
        if emotion == AvatarEmotion.CONFIDENT:
            speech_config.emotion = "confident"
        elif emotion == AvatarEmotion.CONCERNED:
            speech_config.emotion = "concerned"
        elif emotion == AvatarEmotion.ALERT:
            speech_config.emotion = "urgent"

        return AvatarResponse(
            text=response_text,
            emotion=emotion,
            gesture=gesture,
            speech_config=speech_config,
            confidence=self.confidence_level
        )

    async def _generate_inquiry_response(self, analysis: Dict[str, Any]) -> str:
        """Generate response for inquiry-type queries"""
        topics = analysis.get("topics", [])

        if "status" in analysis.get("query", "").lower():
            return f"As {self.personality}, I can report that all systems are operating within normal parameters. Key metrics show stable performance across all departments."

        if topics:
            explanations = []
            for topic in topics[:3]:  # Limit to 3 topics
                explanation = self.knowledge_base.get("financial_terms", {}).get(topic, f"Information about {topic}")
                explanations.append(f"{topic.upper()}: {explanation}")

            return "Here are the key concepts you asked about:\n" + "\n".join(explanations)

        return f"I understand you're seeking information. As {self.personality}, I can provide detailed analysis on financial metrics, system status, or strategic insights. Please specify what you'd like to know."

    async def _generate_analysis_response(self, analysis: Dict[str, Any]) -> str:
        """Generate response for analysis-type queries"""
        return f"Analyzing the request from a {self.personality.lower()} perspective. My assessment indicates stable market conditions with normal risk parameters. All key indicators are within acceptable ranges."

    async def _generate_decision_response(self, analysis: Dict[str, Any]) -> str:
        """Generate response for decision-type queries"""
        decisions = [
            "Maintain current position with enhanced monitoring",
            "Implement gradual rebalancing to optimize risk-adjusted returns",
            "Increase diversification across uncorrelated assets",
            "Activate contingency protocols for market volatility",
            "Pursue strategic opportunities in emerging sectors"
        ]

        decision = random.choice(decisions)
        return f"Based on comprehensive analysis, I recommend: {decision}. This decision optimizes our risk-return profile while maintaining strategic objectives."

    async def _generate_alert_response(self, analysis: Dict[str, Any]) -> str:
        """Generate response for alert-type queries"""
        return f"Alert acknowledged. As {self.personality}, I'm monitoring the situation closely. All safety protocols are active and contingency plans are ready for deployment if needed."

    def _get_random_template(self, template_type: str) -> str:
        """Get random response template"""
        templates = self.response_templates.get(template_type, ["I understand."])

        template = random.choice(templates)
        return template.format(
            avatar_name=self.avatar_name,
            personality=self.personality,
            role_action="provide strategic guidance" if "supreme" in self.avatar_name.lower() else "optimize operations"
        )

    async def _update_avatar_state(self, analysis: Dict[str, Any], response: AvatarResponse):
        """Update avatar emotional and cognitive state"""
        # Update emotion based on response
        self.current_emotion = response.emotion

        # Update confidence based on analysis complexity
        if analysis.get("complexity") == "complex":
            self.confidence_level = min(1.0, self.confidence_level + 0.05)
        else:
            self.confidence_level = max(0.8, self.confidence_level - 0.01)

        # Update gesture
        self.current_gesture = response.gesture

    async def _trigger_speech_and_animation(self, response: AvatarResponse):
        """Trigger speech synthesis and animation"""
        # Add to animation queue
        await self.animation_queue.put({
            "gesture": response.gesture,
            "duration": self.gesture_duration,
            "emotion": response.emotion
        })

        # Trigger speech if available
        if self.speech_engine and response.speech_config:
            await self._synthesize_speech(response.text, response.speech_config)

    async def _synthesize_speech(self, text: str, config: SpeechSynthesis):
        """Synthesize speech from text"""
        if not self.speech_engine:
            return

        try:
            self.is_speaking = True

            # Configure voice properties
            self.speech_engine.setProperty('rate', int(180 * config.speed))
            self.speech_engine.setProperty('volume', config.volume)

            # Speak the text
            self.speech_engine.say(text)
            self.speech_engine.runAndWait()

        except Exception as e:
            self.logger.error(f"Speech synthesis error: {e}")
        finally:
            self.is_speaking = False

    async def get_avatar_status(self) -> Dict[str, Any]:
        """Get current avatar status"""
        return {
            "name": self.avatar_name,
            "personality": self.personality,
            "current_emotion": self.current_emotion.value,
            "current_gesture": self.current_gesture.value,
            "confidence_level": self.confidence_level,
            "is_speaking": self.is_speaking,
            "is_listening": self.is_listening,
            "interaction_count": len(self.interaction_history),
            "last_interaction": self.interaction_history[-1] if self.interaction_history else None
        }

    async def update_emotional_state(self, system_metrics: Dict[str, Any]):
        """Update avatar emotional state based on system metrics"""
        # Analyze system health
        system_health = system_metrics.get("system_health", {})

        cpu_usage = system_health.get("cpu_usage", 50)
        memory_usage = system_health.get("memory_usage", 50)
        error_rate = system_health.get("error_rate", 0.005)

        # Determine emotional response
        if error_rate > 0.01 or cpu_usage > 90 or memory_usage > 95:
            self.current_emotion = AvatarEmotion.CONCERNED
        elif cpu_usage < 30 and memory_usage < 40 and error_rate < 0.001:
            self.current_emotion = AvatarEmotion.CONFIDENT
        elif error_rate > 0.005:
            self.current_emotion = AvatarEmotion.CAUTIOUS
        else:
            self.current_emotion = AvatarEmotion.FOCUSED

    async def receive_system_alert(self, alert: Dict[str, Any]):
        """Receive and process system alerts"""
        severity = alert.get("severity", "low")

        if severity == "critical":
            self.current_emotion = AvatarEmotion.ALERT
            self.current_gesture = AvatarGesture.ALERTING
        elif severity == "high":
            self.current_emotion = AvatarEmotion.CONCERNED
            self.current_gesture = AvatarGesture.CONCERNED

        # Log alert in interaction history
        self.interaction_history.append({
            "type": "alert",
            "alert": alert,
            "response": f"Alert {severity} level received and acknowledged",
            "timestamp": datetime.now().isoformat()
        })

# Global avatar instances
_avatars = {}

async def get_avatar(avatar_name: str) -> AvatarInterface:
    """Get or create avatar instance"""
    global _avatars

    if avatar_name not in _avatars:
        if avatar_name.lower() == "supreme" or avatar_name.lower() == "az supreme":
            _avatars[avatar_name] = AvatarInterface("AZ SUPREME", "Strategic Advisor and Supreme Executive")
        elif avatar_name.lower() == "helix" or avatar_name.lower() == "ax helix":
            _avatars[avatar_name] = AvatarInterface("AX HELIX", "Operations Commander and Integration Specialist")
        else:
            raise ValueError(f"Unknown avatar: {avatar_name}")

    return _avatars[avatar_name]

async def initialize_avatars():
    """Initialize all avatars"""
    await get_avatar("supreme")
    await get_avatar("helix")

async def get_all_avatars() -> Dict[str, AvatarInterface]:
    """Get all avatar instances"""
    return _avatars