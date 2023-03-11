# .SILENT: update-config

IMAGE=torch-classifier:latest
CONTAINER=classifier-torch-framework 

build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

dev:
	docker run -it --rm --gpus all \
	-e PYTHONPATH=/workspace \
	-v ${PWD}:/workspace \
	--name dev-torch \
	$(IMAGE) bash

run-bg:
	docker run -it -d --gpus all \
	-e PYTHONPATH=/workspace \
	-v ${PWD}:/workspace \
	--name $(CONTAINER) \
	$(IMAGE) bash

remove:
	docker container stop $(CONTAINER) && \
	docker container rm $(CONTAINER)

update-config:
	@echo "---UPDATE CONFIG---"
	cp config/clearml.conf ~/
	@echo "---   UPDATED   ---"

ls-models:
	python -c "import timm; print(timm.list_models());" > list-models.txt