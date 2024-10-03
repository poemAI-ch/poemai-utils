import argparse
import json
import logging
import os

import jinja2
from jinja2 import Template
from poemai_utils.openai.ask import Ask
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


PYDANTIC_FORMAT_INSTRUCTIONS_EN = """\
The output should be formatted as a JSON instance that conforms to the JSON
schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description":
"a list of strings", "type": "array", "items": {"type": "string"}}}, "required":
["foo"]}}

the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.
The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:

```
{{schema}}
"""


class Edit(BaseModel):
    replace_start: int
    replace_end: int
    new_lines: list[str]


class Edits(BaseModel):
    edits: list[Edit]
    explanation: str


class AICodeEditor:
    def __init__(self, model, ask=None):
        if ask is not None:
            self.ask = ask
        else:
            _logger.info(f"Using model {model} for code editing")
            self.ask = Ask(
                model=model,
                openai_api_key=os.environ["OPENAI_API_KEY"],
                disable_prompt_log=True,
            )

    @staticmethod
    def load_file(filepath):
        with open(filepath, "r") as file:
            return file.read()

    @classmethod
    def save_file(cls, filepath, content):
        _logger.info(f"Saving file to {filepath}")
        # for l in cls.number_lines(content).split("\n"):
        #     _logger.info(f"{l}")
        # _logger.info(f"End content")

        with open(filepath, "w") as file:
            file.write(content)

    @staticmethod
    def number_lines(file_content):
        lines = file_content.split("\n")
        return "\n".join(
            [f"{str(i + 1).zfill(3)}|{line}" for i, line in enumerate(lines)]
        )

    def construct_prompt(self, context_files, target_file_content, prompt_text):
        formatting_instructions_template = jinja2.Template(
            PYDANTIC_FORMAT_INSTRUCTIONS_EN
        )
        formating_instructions_text = formatting_instructions_template.render(
            schema=Edits.model_json_schema()
        )

        template_text = """
You are an expert software developer tasked with editing a source code file.

{{context}}

And here is the file you need to edit:
{{numbered_file}}
-------------
You are tasked with editing a source code file. The file has line numbers to help specify where and how to edit it. The lines start immediately after the "|" character.
Please provide a JSON-based list of edits that follow these rules:
- Each line in the file has an original number that will remain unchanged.
- You can replace any run of lines with replaced lines. You may add more lines than you remove, or replace them with fewer lines or no lines (effectively deleting them).
- Ensure that the indentation and formatting of the code are maintained. Particularly in python, double-check the indentation levels. New/changed lines should start with the correct number of spaces to fit in the surrounding code.
- Make sure not to do overlapping edits. If you replace lines 5-10, you cannot replace lines 7-12, but rather, replace all lines 5-12 in one edit.

{{formating_instructions_text}}
The user asked you to do the following task:
{{user_prompt}}
        """

        # Generate context descriptions
        context_description = ""
        for i, (filename, content) in enumerate(context_files.items()):
            context_description += f"-------------- Context file {i+1} ({filename}):\n{content}\n---------------------\n\n"

        if context_description != "":
            context_description = f"-------- CONTEXT  -----------\nHere are files relevant for this task::\n{context_description}\n-------- END CONTEXT -----------\n"

        template = Template(template_text)

        # Format the template
        formatted_prompt = template.render(
            context=context_description,
            numbered_file=target_file_content,
            user_prompt=prompt_text,
            formating_instructions_text=formating_instructions_text,
        )
        return formatted_prompt

    def ask_model(self, prompt):
        response = self.ask.ask(prompt, json_mode=True, max_tokens=3000)
        return json.loads(response)

    @staticmethod
    def apply_edits(file_content, edits: Edits):
        lines = {
            i + 1: {"original": line, "inserted": []}
            for i, line in enumerate(file_content.split("\n"))
        }

        for edit in edits.edits:
            _logger.info(f"Applying edit: {edit}")

            start_line_int = edit.replace_start
            end_line_int = edit.replace_end
            for line_int in range(start_line_int, end_line_int + 1):
                if line_int in lines:
                    del lines[line_int]["original"]

            lines[start_line_int]["inserted"] = edit.new_lines

        final_lines = []
        for line_num in sorted(lines):
            line_data = lines[line_num]
            if "original" in line_data:
                final_lines.append(line_data["original"])
            if line_data["inserted"]:
                final_lines.extend(line_data["inserted"])
        return "\n".join(final_lines)

    def run(self, context_files, target_file, prompt_text):
        target_content = self.load_file(target_file)
        numbered_target_content = self.number_lines(target_content)
        context_data = {cf: self.load_file(cf) for cf in context_files}
        model_prompt = self.construct_prompt(
            context_data, numbered_target_content, prompt_text
        )
        _logger.info(f"Prompt generated:")
        for l in model_prompt.split("\n"):
            _logger.info(f"{l}")
        _logger.info(f"End prompt")

        model_response = self.ask_model(model_prompt)
        _logger.info(f"Model response: {model_response}")

        edits = Edits(**model_response)

        updated_content = self.apply_edits(target_content, edits)
        self.save_file(target_file, updated_content)
        _logger.info(f"Edits applied to {target_file}")
        _logger.info(f"Explanation: {edits.explanation}")


def main():
    parser = argparse.ArgumentParser(description="AI-powered code editor")
    parser.add_argument(
        "-c",
        "--context",
        action="append",
        help="Context files to include",
        required=False,
    )
    parser.add_argument("-f", "--file", help="File to edit", required=True)
    parser.add_argument(
        "-p", "--prompt", help="Prompt describing the edit", required=True
    )
    parser.add_argument(
        "--model", help="Model to use for editing", default=None, required=False
    )
    args = parser.parse_args()

    if args.model:
        model = Ask.OPENAI_MODEL.by_model_key(args.model)
    else:
        model = Ask.OPENAI_MODEL.GPT_4_o
    context = args.context or []

    editor = AICodeEditor(model=model)
    editor.run(context, args.file, args.prompt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
