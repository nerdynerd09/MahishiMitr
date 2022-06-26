#Create a ubuntu base image with python 3 installed.
FROM python:3
FROM jjanzic/docker-python3-opencv

COPY . /app
#Set the working directory
WORKDIR /app

#copy all the files
COPY ./requirements.txt requirements.txt

#Install the dependencies
RUN pip3 install -r requirements.txt

COPY . .
#Expose the required port
EXPOSE 5000

#Run the command
ENTRYPOINT [ "python3" ]
# CMD gunicorn app:app
CMD [ "app.py" ]