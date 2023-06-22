from pathlib import Path
import cv2 as cv


def read_data(data_path, eye='L'):
    """Read iris data from given directory.

    :param data_path: Path to directory with iris .jpg images
    :param eye: L or R to select either left or right eye images

    :return: Dictonary with person id as key and list of iris images as value.
    """
    # returns dict id -> List[Img]
    data_path = Path(data_path)

    data = dict()
    for data_dir in data_path.iterdir():
        person_id = data_dir.name
        images_dir = data_dir / eye
        images = [cv.imread(str(path))
                  for path in images_dir.iterdir() if path.suffix == '.jpg']

        if len(images) >= 2:  # at least one train and one test sample
            data[person_id] = images

    return data
