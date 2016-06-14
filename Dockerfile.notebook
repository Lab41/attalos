FROM l41-tensorflow

EXPOSE 8888

RUN pip install --upgrade matplotlib
ADD bootstrap.sh /
WORKDIR /
RUN chmod a+x /bootstrap.sh
CMD ["jupyter", "notebook", "--no-browser", "--ip='*'"]
