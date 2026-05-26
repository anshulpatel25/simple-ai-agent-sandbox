"""Application configuration loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the agent.

    Values are read from environment variables or a ``.env`` file in the
    project root. Each field has a sensible default so the agent works
    out-of-the-box with LM Studio running on localhost.
    """

    # ------------------------------------------------------------------ LLM
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    llm_model: str = "local-model"

    # LLM generation parameters
    llm_temperature: float = 0.0

    # --------------------------------------------------------------- Docker
    docker_image: str = "ubuntu:latest"
    container_timeout: int = 0  # seconds; 0 = no timeout

    # --------------------------------------------------------------- Skills
    skills_base_dir: str = "skills"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Module-level singleton – import this everywhere instead of constructing
# a new Settings() each time to avoid repeated disk reads.
settings = Settings()
