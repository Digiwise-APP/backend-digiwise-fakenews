# backend-digiwise-fakenews


## Installation
Make sure you have poetry installed, follow the instructions from the link to install poetry https://python-poetry.org/docs/

After you have installed poetry.

Install the packages that are required to run the server:
```
poetry install
```

Run the server
```
poetry run uvicorn main:app --port 8080
```

or maybe you already installed Docker in your system, just run following command:
```
docker build -t backend-digiwise-fakenews:latest .
docker run backend-digiwise-fakenews -p 8080:8080 -d
```

Enjoy!
