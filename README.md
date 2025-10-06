# Generic ML

This is a generic machine learning repository that can be used as a template for your own projects.

## How to use this template

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/el-tocino/generic-ml <your-project-name>
    ```

2.  **Rename the package:**

    Rename the `generic_ml` directory to the name of your project's package.

3.  **Update `pyproject.toml`:**

    Change the `name` in `pyproject.toml` to match your project's name.

4.  **Update `mkdocs.yml`:**

    Update the `site_name`, `site_url`, `repo_name`, and `repo_url` to match your project's information.

## Getting Started

1.  **Create and activate the virtual environment:**

    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    uv sync
    ```

## Running the main file

To run the `main.py` file, you can use `uv run`:

```bash
uv run generic_ml/main.py
```

## Using the package

To use the package in your own code, you can import it like this:

```python
import generic_ml

# Your code here
```

## Testing

This project uses `pytest` for testing. To run the tests, use `uv run`:

```bash
uv run pytest
```

## Documentation

This project uses `mkdocs` for documentation. To build and serve the documentation locally, run:

```bash
uv run mkdocs serve
```

This command will start a local server and automatically rebuild the documentation when you make changes.

To build the static documentation site, run:

```bash
uv run mkdocs build
```
