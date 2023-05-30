import json
import typing
from datetime import datetime
from typing import Any, Optional
from collections import defaultdict
from pydantic import BaseModel, validator


class BasicSummary(BaseModel):
    list_names_categories: list[str]
    total_images: int
    total_annotations: int
    total_categories: int
    total_size_kb: float


class DistributionImageSize(BaseModel):
    size_kb: list[int | float]
    width: list[int | float]
    height: list[int | float]
    categories: list[str]


class Summary(BaseModel):
    basic: BasicSummary
    distribution_categories: dict[str, int]
    distribution_image_size: DistributionImageSize
    # scatter_wh_color_category: dict[str, list[list[float]]]


class Info(BaseModel):
    description: str
    version: str = "1.0"
    format_version: str = "1.0"
    url: str
    cvat_url: Optional[str] = None
    cvat_task_id: Optional[int] = None
    contributor: list[str] = []
    type_dataset: Optional[str] = None
    summary: Optional[Summary] = None
    year: Optional[int] | Optional[str] = int(datetime.now().year)
    last_updated: Optional[str] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_created: Optional[str] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @validator("year", allow_reuse=True)
    def year_valid(cls, v):
        if v is None:
            return int(datetime.now().year)
        else:
            return v

    @validator("date_created", allow_reuse=True)
    def date_created_valid(cls, v):
        if v is None:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "T" in v or "Z" in v:
            return datetime.fromisoformat(v).strftime("%Y-%m-%d %H:%M:%S")
        return v

    @validator("last_updated", allow_reuse=True)
    def date_last_updated(cls, v):
        if v is None:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "T" in v:
            return datetime.fromisoformat(v).strftime("%Y-%m-%d %H:%M:%S")
        return v

    # functions
    def update(self, description=None, url=None, year=None, contributor=None):
        if description:
            self.description = description
        if url:
            self.url = url
        if year:
            self.year = year
        if contributor:
            self.contributor = contributor
        self._update_version_and_last_updated()

    def _update_version_and_last_updated(self):
        version_parts = self.version.split(".")
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1])
        new_minor_version = minor_version + 1
        new_version = f"{major_version}.{new_minor_version}"
        self.version = new_version
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Categories(BaseModel):
    id: int
    name: str
    supercategory: str


class License(BaseModel):
    id: int
    name: str
    url: str


class MetadataImage(BaseModel):
    basic: Optional[dict[str, Any]]
    description: Optional[dict[str, Any]]
    features: Optional[dict[str, Any]]


class ImageCoco(BaseModel):
    id: Optional[int]
    license: Optional[int]
    width: Optional[int]
    height: Optional[int]
    url: Optional[str]
    file_name: str
    metadata: Optional[MetadataImage]
    date_captured: Optional[str]


class AnnotationCoco(BaseModel):
    id: Optional[int]
    image_id: Optional[int]
    category_id: int
    segmentation: Optional[list[list[float]]] = []
    area: Optional[float] = 0.0
    bbox: Optional[list[float]] = []
    iscrowd: Optional[int] = 0
    attributes: Optional[typing.Dict[str, typing.Any]] = {}


class CocoFormat(BaseModel):
    info: Info
    licenses: Optional[list[License]] = [License(id=0, name="sample", url="")]
    categories: Optional[list[Categories]] = []
    images: Optional[list[ImageCoco]] = []
    annotations: Optional[list[AnnotationCoco]] = []

    @validator("licenses")
    def licenses_valid(cls, v):
        if v is None:
            return [License(id=0, name="sample", url="")]
        return v

    @validator("categories")
    def categories_valid(cls, v):
        if v is None:
            return []
        return v

    @validator("images")
    def images_valid(cls, v):
        if v is None:
            return []
        return v

    @validator("annotations")
    def annotations_valid(cls, v):
        if v is None:
            return []
        return v

    # functions
    def get_img_urls_by_category(self) -> dict[str, list[str]]:
        idcat2lblcat = {cat.id: cat.name for cat in self.categories}
        d_data = self.dict_annotations_by_img_id()
        d_image = self.dict_images_by_id()
        d_download = defaultdict(list)
        for img_id, anns in d_data.items():
            ann = anns[0]
            label_name = idcat2lblcat[ann.category_id]

            # if len(d_download[label_name]) > 5:
            #     continue

            d_download[label_name].append(d_image[img_id].url)
        return d_download

    def dict_images_by_id(self) -> dict[int, ImageCoco]:
        return {img.id: img for img in self.images}

    def get_list_images_id(self):
        return [img.id for img in self.images]

    def get_annotations_by_image_id(self, image_id: int) -> list[AnnotationCoco]:
        annotations = [anno for anno in self.annotations if anno.image_id == image_id]
        return annotations

    def get_list_annotations_id(self):
        return [anno.id for anno in self.annotations]

    def dict_annotations_by_img_id(self) -> dict[int, list[AnnotationCoco]]:
        dict_annotations = {}
        for img in self.images:
            dict_annotations[img.id] = self.get_annotations_by_image_id(img.id)
        return dict_annotations

    def list_categories_id(self):
        return {cat.id: cat for cat in self.categories}

    def cat_id_to_name(self):
        return {cat.id: cat.name for cat in self.categories}

    def name_to_cat_id(self):
        return {cat.name: cat.id for cat in self.categories}

    def ann_id_to_index(self):
        return {ann.id: i for i, ann in enumerate(self.annotations)}
