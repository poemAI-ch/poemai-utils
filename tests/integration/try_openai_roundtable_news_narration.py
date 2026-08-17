#!/usr/bin/env python3
"""Generate a live MP3 using settings equivalent to the staging narration profile.

Run manually from the repository root:

    python tests/integration/try_openai_roundtable_news_narration.py

Requires ``OPENAI_API_KEY``. This script calls the live OpenAI API and is
intentionally not collected by pytest.
"""

import argparse
import logging
import os
from pathlib import Path

from poemai_utils.audio import (
    OPENAI_SPEECH_MODEL,
    OpenAITextToSpeech,
    SpeechOutputFormat,
    TextToSpeechRequest,
)

_logger = logging.getLogger(__name__)

NEWS_ARTICLE = """Weltrekord in Brunnen aufgestellt: Robin Steiner flog mehr als 28 Meter hoch.

An der Windweek in Brunnen glückte der Weltrekordversuch des Berners Robin Steiner in der Disziplin Blobbing hervorragend."""

SWISS_STANDARD_GERMAN_INSTRUCTIONS = """Speak German with a natural Swiss Standard German accent
(Schweizer Hochdeutsch), as spoken in German-speaking Switzerland.
Do not use a Germany-German accent.
Do not speak Swiss German dialect.
Use natural Swiss pronunciation, rhythm and intonation.
Speak clearly in a calm, neutral presentation style."""

DEFAULT_OUTPUT_PATH = (
    Path(__file__).with_name("artifacts") / "robin_steiner_blobbing_news.mp3"
)


def _looks_like_mp3(audio_bytes):
    return audio_bytes.startswith(b"ID3") or (
        len(audio_bytes) >= 2
        and audio_bytes[0] == 0xFF
        and audio_bytes[1] & 0xE0 == 0xE0
    )


def generate_news_narration(output_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY to run this live integration test.")

    request = TextToSpeechRequest(
        NEWS_ARTICLE,
        model=OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS,
        voice="cedar",
        output_format=SpeechOutputFormat.MP3,
        instructions=SWISS_STANDARD_GERMAN_INSTRUCTIONS,
        speed=1.0,
    )
    result = OpenAITextToSpeech(api_key=api_key).synthesize(request)

    if result.provider != "openai":
        raise RuntimeError(f"Unexpected provider: {result.provider}")
    if result.model_key != OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS.model_key:
        raise RuntimeError(f"Unexpected model: {result.model_key}")
    if result.voice != "cedar":
        raise RuntimeError(f"Unexpected voice: {result.voice}")
    if result.output_format is not SpeechOutputFormat.MP3:
        raise RuntimeError(f"Unexpected output format: {result.output_format}")
    if result.content_type != "audio/mpeg" or result.file_extension != ".mp3":
        raise RuntimeError("Unexpected MP3 metadata in text-to-speech result")
    if not _looks_like_mp3(result.audio_bytes):
        raise RuntimeError("OpenAI response does not look like an MP3 file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.audio_bytes)
    _logger.info(
        "Generated live OpenAI narration; open this file to listen: %s (%d bytes, request_id=%s)",
        output_path.resolve(),
        len(result.audio_bytes),
        result.provider_request_id,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="MP3 output path (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_news_narration(args.output)


if __name__ == "__main__":
    main()
