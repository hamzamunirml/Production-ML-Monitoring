@echo off
echo ============================================================
echo Starting Bank Marketing Prediction API
echo ============================================================
echo.
echo Installing dependencies...
pip install -r requirements_api.txt
echo.
echo Starting FastAPI server...
echo API will be available at: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo.
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
pause