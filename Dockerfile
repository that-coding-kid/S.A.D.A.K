FROM python:3.10.11

# Expose port you want your app on
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app code and set working directory
COPY analysis analysis
COPY structures structures
COPY configure configure
COPY images images
COPY locales locales
COPY scripts scripts
COPY utils utils

COPY weights weights
COPY helper.py helper.py
COPY app.py app.py
COPY assets assets
COPY settings.py settings.py
COPY traffic_data.csv traffic_data.csv



WORKDIR ./

# Run
ENTRYPOINT [“streamlit”, “run”, “app.py”, “–server.port=8080”, “–server.address=0.0.0.0”]