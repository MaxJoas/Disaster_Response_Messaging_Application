FROM python:3.7.6

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' model-app-user

WORKDIR /app

ENV FLASK_APP run.py
ENV NLTK_DATA /app/nltk_data/ 

# Install requirements, including from Gemfury
ADD . /app
ADD . $NLTK_DATA

RUN pip install --upgrade pip
RUN pip install poetry
RUN make poetry-install
RUN make nltk-resources

RUN chmod +x run.sh
RUN chown -R model-app-user:model-app-user ./

USER model-app-user

EXPOSE 5000

CMD ["bash", "./run.sh"]