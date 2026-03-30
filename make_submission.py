import os
import zipfile


def main():
    print("Packing it all up...")
    submission_filename = "submission.zip"

    ignore_dirs = {"data", "__pycache__"}

    with zipfile.ZipFile(submission_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d not in ignore_dirs
            ]

            for file in files:
                if file.startswith(".") or file == submission_filename or file.endswith(".pdf"):
                    continue

                file_path = os.path.join(root, file)
                archive_path = os.path.relpath(file_path, ".")

                zipf.write(file_path, arcname=archive_path)

    print(f"\n'{submission_filename}' is ready for upload!")


if __name__ == "__main__":
    main()
