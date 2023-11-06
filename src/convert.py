import supervisely as sly
import os
import glob
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.json import load_json_file
from supervisely.io.fs import get_file_name, get_file_name_with_ext
import shutil

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    dataset_path = os.path.join("Iraqi Money","Project-1-JPG")
    ann_path = os.path.join("Iraqi Money","Project-1-JPG","create_ml_labels.json")
    batch_size = 30
    ds_name = "ds"


    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        image_name = get_file_name_with_ext(image_path)
        ann_data = name_to_data[image_name]
        for curr_ann_data in ann_data:
            obj_class = meta.get_obj_class(curr_ann_data["label"])
            coords = curr_ann_data["coordinates"]
            left = int(coords["x"]) - int(coords["width"]) / 2
            top = int(coords["y"]) - int(coords["height"]) / 2
            right = left + int(coords["width"])
            bottom = top + int(coords["height"])
            rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
            label = sly.Label(rect, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    ar250 = sly.ObjClass("250ar", sly.Rectangle)
    en250 = sly.ObjClass("250en", sly.Rectangle)
    ar500 = sly.ObjClass("500ar", sly.Rectangle)
    en500 = sly.ObjClass("500en", sly.Rectangle)
    ar1000 = sly.ObjClass("1000ar", sly.Rectangle)
    en1000 = sly.ObjClass("1000en", sly.Rectangle)
    ar5000 = sly.ObjClass("5000ar", sly.Rectangle)
    en5000 = sly.ObjClass("5000en", sly.Rectangle)
    ar10000 = sly.ObjClass("10000ar", sly.Rectangle)
    en10000 = sly.ObjClass("10000en", sly.Rectangle)
    ar25000 = sly.ObjClass("25000ar", sly.Rectangle)
    en25000 = sly.ObjClass("25000en", sly.Rectangle)
    ar50000 = sly.ObjClass("50000ar", sly.Rectangle)
    en50000 = sly.ObjClass("50000en", sly.Rectangle)


    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[
            ar250,
            en250,
            ar500,
            en500,
            ar1000,
            en1000,
            ar5000,
            en5000,
            ar10000,
            en10000,
            ar25000,
            en25000,
            ar50000,
            en50000,
        ]
    )
    api.project.update_meta(project.id, meta.to_json())

    name_to_data = {}

    ann = load_json_file(ann_path)
    for curr_ann in ann:
        name_to_data[curr_ann["imagename"].split("/")[-1]] = curr_ann["annotations"]

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_pathes = glob.glob(dataset_path + "/*/*.jpg")

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

    for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
        img_names_batch = [get_file_name_with_ext(im_path) for im_path in img_pathes_batch]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(image_path) for image_path in img_pathes_batch]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(img_names_batch))

    return project
