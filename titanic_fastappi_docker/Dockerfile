#Step 1: Base image (lightweight python)
FROM python:3.10-slim

#Step 2: Set working directory inside the container
WORKDIR /app

#Step 3: Copy all libraries from requirements.txt file
COPY requirements.txt .

#Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Step 5: Copy application Code
COPY fastapi_app.py .
COPY artifacts/ artifacts/

#Step 6 Expose API Port
EXPOSE 8000

#Step 7: Start FastAPI using Uvicorn
CMD ["uvicorn","fastapi_app:app", "--host","0.0.0.0","--port","8000"]