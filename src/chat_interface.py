"""Gradio interface for the AskBio evidence-first tutor."""

from __future__ import annotations

import logging
from functools import lru_cache

import gradio as gr

from askbio import (
    EXPLANATION_LEVELS,
    SUPPORTED_LANGUAGES,
    TUTOR_MODES,
    AskBio,
    AskBioError,
)
from progress import LearningProgressStore

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_askbio() -> AskBio:
    """Initialize documents, index, and models only on the first question."""
    return AskBio()


@lru_cache(maxsize=1)
def get_progress_store() -> LearningProgressStore:
    return LearningProgressStore()


def chat_with_askbio(
    message: str,
    history: list | None = None,
    mode: str = "Learn",
    level: str = "Intermediate",
    language: str = "English",
) -> str:
    """Answer a question and render the evidence returned by retrieval."""
    if not message or not message.strip():
        return "Please enter a biology question."

    try:
        response = get_askbio().ask_with_sources(
            message,
            mode=mode,
            level=level,
            history=history,
            language=language,
        )
        get_progress_store().record(
            question=message,
            mode=mode,
            level=level,
            response=response,
        )
        return response.to_markdown()
    except AskBioError as exc:
        LOGGER.warning("AskBio setup error: %s", exc)
        return f"**Setup needed:** {exc}"
    except Exception:
        LOGGER.exception("AskBio failed to process a question")
        return "AskBio could not process that question. Please try again."


def build_interface() -> gr.ChatInterface:
    """Construct the web interface without loading the inference backend."""
    return gr.ChatInterface(
        chat_with_askbio,
        chatbot=gr.Chatbot(height=700),
        textbox=gr.Textbox(
            placeholder="Ask a question about your biology textbook…",
            container=False,
            scale=7,
        ),
        additional_inputs=[
            gr.Dropdown(
                choices=list(TUTOR_MODES),
                value="Learn",
                label="Learning mode",
            ),
            gr.Dropdown(
                choices=list(EXPLANATION_LEVELS),
                value="Intermediate",
                label="Explanation level",
            ),
            gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES),
                value="English",
                label="Response language",
            ),
        ],
        additional_inputs_accordion="Tutor settings",
        title="AskBio",
        description=(
            "An evidence-first biology tutor. Answers include the textbook passages "
            "used as evidence."
        ),
    )


iface = build_interface()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    iface.launch()
