import sys, uvicorn
sys.path.insert(0, r"C:\Users\Administrator\securtyagent")
from app.main import app
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
