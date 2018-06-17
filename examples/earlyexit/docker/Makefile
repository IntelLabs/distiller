IMAGE=earlyexit

.PHONY: build run

BUILD_ARGUMENTS=
RUN_ARGUMENTS=
ifdef http_proxy
	BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
	RUN_ARGUMENTS+=--env http_proxy=$(http_proxy)
endif

ifdef https_proxy
	BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
	RUN_ARGUMENTS+=--env https_proxy=$(https_proxy)
endif

CONTEXT = $(realpath ..)

# use nvidia-docker if it is available
DOCKER := $(shell command -v nvidia-docker 2> /dev/null)
ifndef DOCKER
	DOCKER = docker
endif

clean:
	@rm -f .*.swp .*.swo
	@rm -f *.pyc
	@rm -rf ${CONTEXT}/docker/private-aeon

build: 
	${DOCKER} build -f=Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT}

push:
	${DOCKER} push ${IMAGE}

interact:
	kubectl create -f interactive_pod.yaml

