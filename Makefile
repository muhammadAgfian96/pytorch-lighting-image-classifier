# .SILENT: update-config

IMAGE=torch-classifier:latest
CONTAINER=dev-pl-timm-training

build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

dev:
	docker run -it --rm --gpus all \
	-e PYTHONPATH=/workspace \
	-v ${PWD}:/workspace \
	--shm-size=8g \
	-v /home/binshoadmin/clearml.conf:/root/clearml.conf \
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

exec:
	docker exec -it $(CONTAINER) bash