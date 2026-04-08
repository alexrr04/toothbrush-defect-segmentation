import os
import zipfile


def _load_selected_models(weights_file_path):
    if not os.path.exists(weights_file_path):
        raise FileNotFoundError(
            f"Missing {weights_file_path}. Create it with one model path per line."
        )

    selected_paths = []
    with open(weights_file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            normalized = os.path.normpath(line)
            if os.path.isabs(normalized):
                raise ValueError(
                    "Use workspace-relative paths in ensemble_weights.txt, "
                    f"found absolute path: {line}"
                )
            selected_paths.append(normalized)

    if not selected_paths:
        raise ValueError(
            f"No model paths found in {weights_file_path}. Add one .pth path per line."
        )

    missing = [p for p in selected_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "These model paths from ensemble_weights.txt do not exist:\n"
            + "\n".join(missing)
        )

    return set(selected_paths)


def main():
    print("Packing it all up...")
    submission_filename = "submission.zip"
    weights_file = os.path.normpath("trained_models/ensemble_weights.txt")
    selected_models = _load_selected_models(weights_file)

    ignore_dirs = {"data", "__pycache__"}

    with zipfile.ZipFile(submission_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d not in ignore_dirs
            ]

            for file in files:
                if (
                    file.startswith(".")
                    or file == submission_filename
                    or file.endswith(".pdf")
                ):
                    continue

                file_path = os.path.join(root, file)
                archive_path = os.path.normpath(os.path.relpath(file_path, "."))

                # Include only selected model checkpoints from trained_models.
                if archive_path.endswith(".pth") and archive_path.startswith(
                    "trained_models"
                ):
                    if archive_path not in selected_models:
                        continue

                # Keep ensemble definition file for reproducibility.
                if archive_path == weights_file or archive_path in selected_models:
                    zipf.write(file_path, arcname=archive_path)
                    continue

                zipf.write(file_path, arcname=archive_path)

    print(f"\n'{submission_filename}' is ready for upload!")
    print("Included model checkpoints:")
    for model_path in sorted(selected_models):
        print(f"  - {model_path}")


if __name__ == "__main__":
    main()
