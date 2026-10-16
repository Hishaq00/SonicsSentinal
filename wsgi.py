"""Production entry point.

Windows:  waitress-serve --host 0.0.0.0 --port 8000 wsgi:app
Linux:    gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000 wsgi:app
"""
from dotenv import load_dotenv

load_dotenv()

from src import create_app  # noqa: E402

app = create_app()
