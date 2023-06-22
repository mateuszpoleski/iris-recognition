from tqdm import tqdm

from data import read_data
from normalize import normalize_image
from encode import encode_image
from match import compare_codes


def create_iris_codes(train_data):
    """Creates iris code for each eye image and map it to person id. 

    :param train_data: Dictionary with person id as key and eye image as value

    :return: Iris code for each person in dataset. 
    """
    codes = dict()

    for id, image in train_data.items():
        normalized = normalize_image(image)
        code = encode_image(normalized)
        codes[id] = code

    return codes


def evaluate(test_data, iris_templates):
    """Evaluate performance by matching iris images to existing iris code base.

    Performance is measured by accuracy of determining correct person id from iris image.

    :param test_data: Dictionary with person id as key and eye images as value
    :param iris_templates: Dictionary with database of previously created iris codes for all people in test_data

    :return: Accuracy of matching.
    """
    correct = wrong = 0

    for id, images in tqdm(test_data.items()):
        for image in images:
            normalized = normalize_image(image)
            code = encode_image(normalized)

            dists = []
            for template_id, template_code in iris_templates.items():
                dist = compare_codes(code, template_code)
                dists.append((dist, template_id))

            min_dist = min(dists)

            if id == min_dist[1]:
                correct += 1
            else:
                wrong += 1

    return correct / (correct + wrong)


if __name__ == '__main__':
    iris_dataset_path = '../CASIA-Iris-Interval'

    data = read_data(iris_dataset_path, eye='L')
    train_data = {id: images[0] for id, images in data.items()}
    test_data = {id: images[1:] for id, images in data.items()}

    print('preparing iris codes database')
    iris_codes = create_iris_codes(train_data)
    print('evaluating accuracy')
    acc = evaluate(test_data, iris_codes)
    print(f'matching accuracy: {acc}')
