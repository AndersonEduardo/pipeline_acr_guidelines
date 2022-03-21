FROM python:3.8-slim

# RUN useradd --create-home --shell /bin/bash datascientist

RUN mkdir /experiment

# WORKDIR /home/datascientist
WORKDIR /experiment

COPY requirements.txt ./

RUN apt-get update
RUN pip install --upgrade pip
RUN apt-get install ffmpeg libsm6 libxext6 poppler-utils tesseract-ocr \
libtesseract-dev ghostscript apt-utils build-essential gcc nano -y
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir tmpdir
# RUN chown datascientist:datascientist -R ./tmpdir/
# RUN chmod -R 777 ./tmpdir/

# USER datascientist

ENV JAVA_FOLDER java-se-8u41-ri
ENV JVM_ROOT /usr/lib/jvm
ENV JAVA_PKG_NAME openjdk-8u41-b04-linux-x64-14_jan_2020.tar.gz
ENV JAVA_TAR_GZ_URL https://download.java.net/openjdk/jdk8u41/ri/$JAVA_PKG_NAME

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*    && \
    apt-get clean                                                               && \
    apt-get autoremove                                                          && \
    echo Downloading $JAVA_TAR_GZ_URL                                           && \
    wget -q $JAVA_TAR_GZ_URL                                                    && \
    tar -xvf $JAVA_PKG_NAME                                                     && \
    rm $JAVA_PKG_NAME                                                           && \
    mkdir -p /usr/lib/jvm                                                       && \
    mv ./$JAVA_FOLDER $JVM_ROOT                                                 && \
    update-alternatives --install /usr/bin/java java $JVM_ROOT/$JAVA_FOLDER/bin/java 1        && \
    update-alternatives --install /usr/bin/javac javac $JVM_ROOT/$JAVA_FOLDER/bin/javac 1     && \
    java -version

COPY . .

CMD ["bash"]