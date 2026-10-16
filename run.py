"""Development server:  python run.py   ->  http://127.0.0.1:5000"""
import os

from dotenv import load_dotenv

load_dotenv()

from src import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"SonicSentinel AI running on http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=debug)
