import argparse
import logging

from poemai_utils.openai.ask import Ask

_logger = logging.getLogger(__name__)


def main():

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Converts an HTML file to a markdown file."
    )

    # Add arguments
    parser.add_argument("input_file", help="The input file to process.")
    parser.add_argument("-o", "--output", help="The output file.", required=False)

    # Parse the arguments
    args = parser.parse_args()

    if args.output is None:
        output_file = args.input_file.replace(".html", ".md")

    _logger.info(
        f"Converting {args.input_file} to markdown. Output will be saved to {output_file}"
    )

    with open(args.input_file, "r") as f:
        html = f.read()

    CHUNK_SIZE = 2000

    # Split the html into chunks

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text()

    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < CHUNK_SIZE:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + "."

    if current_chunk:
        chunks.append(current_chunk)

    ask = Ask(model=Ask.OPENAI_MODEL.GPT_4_o_2024_05_13)

    output_markdown = ""

    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        _logger.info(f"Processing chunk {i+1}/{num_chunks}")
        previous_output = ""
        if output_markdown != "":
            previous_output = "\n".join(output_markdown.split("\n")[-5:])
            previous_output = (
                "--- MARKDOWN SO FAR ---\n"
                + previous_output
                + "\n--- MARKDOWN SO FAR ---\n"
            )

        markdown_so_far_instructions = ""
        if previous_output != "":
            markdown_so_far_instructions = "The MARKDOWN SO FAR section shows what has already been formatted in markdown. Do not include it again, your output will be appended to it."

        prompt = f"""   
--- CONTEXT ---
--- HTML SNIPPET ---
{chunk}
--- END HTML SNIPPET ---
{previous_output}

---- END CONTEXT ----
Translate the text snippet. The snippet was extracted from a larger html document. {markdown_so_far_instructions}
Remove advertisements, comments, and other irrelevant content. Make sure you use the text as-is, just adapt the formatting and structure the text into sections. Try to identify title, headings and subheadings in the text and format them accordingly.
Do not give any comments on this instructions, only give the markdown result. The markdown should be formatted correctly.
"""
        result = ask.ask(prompt, max_tokens=3000)
        result = result.replace("```markdown", "")
        result = result.replace("```", "")
        output_markdown += "\n" + result

        _logger.info(f"Output so far: {output_markdown}")

    try:
        with open(output_file, "w") as f:
            f.write(output_markdown)
            _logger.info(f"Output saved to {output_file}")
    except Exception as e:
        _logger.error(f"Error saving output to {output_file}: {e}")


if __name__ == "__main__":
    main()
