FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY Artifacts/model.pkl /code/app/Artifacts/

COPY ./app /code/app

# Expose the port Streamlit runs on
EXPOSE 8000

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]