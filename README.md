# Backend

This directory contains the backend for the Buraco card game.

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Python 3.10+](https://www.python.org/downloads/)
*   [uv](https://github.com/astral-sh/uv)

## Installation

1.  **Create a virtual environment:**

    ```bash
    uv venv
    ```

2.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    uv pip install -r requirements.txt
    ```

## Running the Application

To run the FastAPI application, use the following command:

```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`.

## Running Tests

To run the tests, use the following command:

```bash
uv run pytest
```