"""Backward-compatible imports for the text-to-speech API."""

from poemai_utils.audio.contracts import TextToSpeechRequest, TextToSpeechResult
from poemai_utils.audio.openai_text_to_speech import OpenAITextToSpeech

__all__ = ["TextToSpeechRequest", "TextToSpeechResult", "OpenAITextToSpeech"]
