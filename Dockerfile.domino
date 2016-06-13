FROM l41-tensorflow
MAINTAINER Alex Gude <agude@iqt.org>

# Set up the Ubuntu User that Domino Expects
RUN addgroup dominouser --gid 4100 && \
    addgroup ubuntu && \
    adduser ubuntu --disabled-password --ingroup dominouser
