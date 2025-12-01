"""
A2A (Agent-to-Agent) Orchestrator for CIO-Agent.

Manages communication between the Green Agent (CIO-Agent/Evaluator)
and White/Purple Agents (test agents) using the A2A Protocol.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Callable, Optional

import structlog
from pydantic import BaseModel, Field

from cio_agent.models import (
    Task,
    AgentResponse,
    DebateRebuttal,
    A2AMessage,
    A2AMessageType,
)

logger = structlog.get_logger()


class AgentConfig(BaseModel):
    """Configuration for an agent connection."""
    agent_id: str
    endpoint_url: str
    timeout_seconds: int = 1800  # 30 minutes default
    model: str = "gpt-4o"


class A2AOrchestrator:
    """
    Orchestrates communication between CIO-Agent and test agents.

    Responsibilities:
    - Send task assignments to agents
    - Receive and parse agent responses
    - Send adversarial challenges
    - Collect rebuttals
    - Track message timing and metadata
    """

    def __init__(
        self,
        cio_agent_id: str = "cio-agent-green",
        message_handler: Optional[Callable] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            cio_agent_id: ID of the CIO-Agent (Green Agent)
            message_handler: Optional custom message handler
        """
        self.cio_agent_id = cio_agent_id
        self.message_handler = message_handler
        self.message_log: list[A2AMessage] = []
        self._pending_responses: dict[str, asyncio.Future] = {}

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        return f"msg_{uuid.uuid4().hex[:12]}"

    def _log_message(self, message: A2AMessage) -> None:
        """Log a message for audit trail."""
        self.message_log.append(message)
        logger.info(
            "a2a_message",
            type=message.message_type.value,
            sender=message.sender_id,
            receiver=message.receiver_id,
            payload_keys=list(message.payload.keys()),
        )

    async def send_task_assignment(
        self,
        agent_id: str,
        task: Task,
    ) -> A2AMessage:
        """
        Send a task assignment to an agent.

        Args:
            agent_id: Target agent ID
            task: Task to assign

        Returns:
            The sent A2AMessage
        """
        message = A2AMessage.task_assignment(
            sender_id=self.cio_agent_id,
            receiver_id=agent_id,
            task=task,
        )

        self._log_message(message)

        if self.message_handler:
            await self.message_handler(message)

        logger.info(
            "task_assigned",
            agent_id=agent_id,
            task_id=task.question_id,
            deadline_seconds=task.deadline_seconds,
        )

        return message

    async def receive_task_response(
        self,
        agent_id: str,
        timeout_seconds: int = 1800,
    ) -> Optional[AgentResponse]:
        """
        Wait for and receive a task response from an agent.

        Args:
            agent_id: Agent ID to receive from
            timeout_seconds: Maximum wait time

        Returns:
            AgentResponse or None if timeout
        """
        response_key = f"response_{agent_id}"
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[response_key] = future

        try:
            response = await asyncio.wait_for(future, timeout=timeout_seconds)
            return response
        except asyncio.TimeoutError:
            logger.warning("response_timeout", agent_id=agent_id, timeout=timeout_seconds)
            return None
        finally:
            self._pending_responses.pop(response_key, None)

    def deliver_response(
        self,
        agent_id: str,
        response: AgentResponse,
    ) -> None:
        """
        Deliver a response to a waiting receiver.

        Called by message handler when response is received.

        Args:
            agent_id: Agent ID that sent the response
            response: The response to deliver
        """
        response_key = f"response_{agent_id}"
        if response_key in self._pending_responses:
            future = self._pending_responses[response_key]
            if not future.done():
                future.set_result(response)

    async def send_challenge(
        self,
        agent_id: str,
        task_id: str,
        counter_argument: str,
    ) -> A2AMessage:
        """
        Send an adversarial challenge to an agent.

        Args:
            agent_id: Target agent ID
            task_id: ID of the task being challenged
            counter_argument: The counter-argument/challenge

        Returns:
            The sent A2AMessage
        """
        message = A2AMessage.challenge(
            sender_id=self.cio_agent_id,
            receiver_id=agent_id,
            task_id=task_id,
            counter_argument=counter_argument,
        )

        self._log_message(message)

        if self.message_handler:
            await self.message_handler(message)

        logger.info(
            "challenge_sent",
            agent_id=agent_id,
            task_id=task_id,
        )

        return message

    async def receive_rebuttal(
        self,
        agent_id: str,
        timeout_seconds: int = 600,  # 10 minutes for rebuttal
    ) -> Optional[DebateRebuttal]:
        """
        Wait for and receive a rebuttal from an agent.

        Args:
            agent_id: Agent ID to receive from
            timeout_seconds: Maximum wait time

        Returns:
            DebateRebuttal or None if timeout
        """
        rebuttal_key = f"rebuttal_{agent_id}"
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[rebuttal_key] = future

        try:
            rebuttal = await asyncio.wait_for(future, timeout=timeout_seconds)
            return rebuttal
        except asyncio.TimeoutError:
            logger.warning("rebuttal_timeout", agent_id=agent_id, timeout=timeout_seconds)
            return None
        finally:
            self._pending_responses.pop(rebuttal_key, None)

    def deliver_rebuttal(
        self,
        agent_id: str,
        rebuttal: DebateRebuttal,
    ) -> None:
        """
        Deliver a rebuttal to a waiting receiver.

        Args:
            agent_id: Agent ID that sent the rebuttal
            rebuttal: The rebuttal to deliver
        """
        rebuttal_key = f"rebuttal_{agent_id}"
        if rebuttal_key in self._pending_responses:
            future = self._pending_responses[rebuttal_key]
            if not future.done():
                future.set_result(rebuttal)

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[A2AMessageType] = None,
    ) -> list[A2AMessage]:
        """
        Get filtered message history.

        Args:
            agent_id: Optional filter by agent ID
            message_type: Optional filter by message type

        Returns:
            Filtered list of messages
        """
        messages = self.message_log

        if agent_id:
            messages = [
                m for m in messages
                if m.sender_id == agent_id or m.receiver_id == agent_id
            ]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages


