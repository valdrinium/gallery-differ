# set base image (host OS)
FROM python:3.9

# set the working directory in the container
WORKDIR /code

# copy everything to the working directory
COPY . .

# install dependencies
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "sh", "-c", "FLASK_APP=src/server.py FLASK_ENV=development flask run --host=0.0.0.0 --port 80" ]
