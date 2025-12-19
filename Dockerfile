FROM python:3.9-slim

WORKDIR /app

COPY MLProject /app/MLProject
COPY MLProject/conda.yaml /app/conda.yaml

RUN pip install --upgrade pip \
    && pip install mlflow scikit-learn pandas numpy

CMD ["mlflow", "run", "MLProject", "--env-manager=local"]