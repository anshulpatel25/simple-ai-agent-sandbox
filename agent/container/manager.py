"""Docker container lifecycle management.

Each CLI session owns exactly one container. ``DockerContainerManager``
acts as a context manager so the container is always cleaned up – even
when the session ends with an exception or Ctrl-C.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import TracebackType
from typing import Optional

import docker
import docker.errors
from docker.models.containers import Container

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Result of running a command inside the container."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    def combined_output(self) -> str:
        """Return stdout and stderr merged, for easy display."""
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.strip()}")
        return "\n".join(parts) if parts else "(no output)"


class DockerContainerManager:
    """Manages the lifecycle of a single Ubuntu container per agent session.

    Usage::

        with DockerContainerManager(image="ubuntu:latest") as manager:
            result = manager.exec("echo hello")
            print(result.stdout)
    """

    def __init__(self, image: str = "ubuntu:latest") -> None:
        self._image = image
        self._client: docker.DockerClient = docker.from_env()
        self._container: Optional[Container] = None

    # ---------------------------------------------------------------- setup
    def create(self) -> None:
        """Pull the image if necessary and start a detached container."""
        logger.info("Pulling image %s (if not cached)…", self._image)
        try:
            self._client.images.get(self._image)
        except docker.errors.ImageNotFound:
            logger.info("Image not found locally, pulling…")
            self._client.images.pull(self._image)

        logger.info("Creating container from %s…", self._image)
        self._container = self._client.containers.run(
            image=self._image,
            command="sleep infinity",  # keep the container alive
            detach=True,
            tty=True,
            stdin_open=True,
            remove=False,  # we remove explicitly in destroy()
        )
        logger.info("Container started: %s", self._container.short_id)

    # -------------------------------------------------------------- teardown
    def destroy(self) -> None:
        """Stop and remove the container, ignoring errors if already gone."""
        if self._container is None:
            return
        try:
            self._container.stop(timeout=5)
            self._container.remove(force=True)
            logger.info("Container %s removed.", self._container.short_id)
        except docker.errors.NotFound:
            logger.debug("Container already gone, nothing to remove.")
        except docker.errors.APIError as exc:
            logger.warning("Error removing container: %s", exc)
        finally:
            self._container = None

    # -------------------------------------------------------------- execution
    def exec(self, command: str) -> ExecResult:
        """Run *command* inside the container and return the result.

        Args:
            command: A shell command string, e.g. ``"ls -la /tmp"``.

        Returns:
            An :class:`ExecResult` with exit code, stdout, and stderr.

        Raises:
            RuntimeError: If the container has not been created yet.
        """
        if self._container is None:
            raise RuntimeError(
                "Container is not running. Call create() or use as a context manager."
            )

        logger.debug("Executing in container: %s", command)
        exec_result = self._container.exec_run(
            cmd=["bash", "-c", command],
            stdout=True,
            stderr=True,
            demux=True,  # separate stdout/stderr streams
        )

        exit_code: int = exec_result.exit_code
        raw_stdout, raw_stderr = exec_result.output  # type: ignore[misc]

        return ExecResult(
            exit_code=exit_code,
            stdout=(raw_stdout or b"").decode("utf-8", errors="replace"),
            stderr=(raw_stderr or b"").decode("utf-8", errors="replace"),
        )

    @property
    def container_id(self) -> Optional[str]:
        """Short Docker container ID, or ``None`` if not started."""
        return self._container.short_id if self._container else None

    # --------------------------------------------------- context manager API
    def __enter__(self) -> "DockerContainerManager":
        self.create()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.destroy()
