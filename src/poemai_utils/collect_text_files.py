import argparse
import os


def collect_source_files(directory, output_file, extensions=None):
    if extensions is None:
        extensions = [".py"]  # Default to Python files if no extension is specified

    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    outfile.write(f"{'=' * 80}\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write(f"{'=' * 80}\n\n")

                    with open(file_path, "r", encoding="utf-8") as infile:
                        content = infile.read()
                        outfile.write(content)
                        outfile.write("\n\n")

    print(f"All source files have been written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect source code files from a directory and its subdirectories into a single text file."
    )
    parser.add_argument(
        "path", type=str, help="Path to the directory containing the source code"
    )
    parser.add_argument("output", type=str, help="Path to the output text file")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".py"],
        help="List of file extensions to include. Default is .py",
    )

    args = parser.parse_args()

    collect_source_files(args.path, args.output, args.extensions)


if __name__ == "__main__":
    main()
