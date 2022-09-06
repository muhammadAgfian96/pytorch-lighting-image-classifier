# .SILENT: update-config

IMAGE=torch-classifier:1.12
CONTAINER=classifier-torch-framework 

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