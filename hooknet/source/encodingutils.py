
from pathlib import Path
from tensorflow.python.keras.models import Model
from hooknet.source.model import HookNet
import numpy as np
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation

def create_hooknet_encoder(hooknet: HookNet):
    encoding_layer = hooknet.get_layer('target-branchbottle').output
    return Model(hooknet.inputs, encoding_layer)

def compute_encodings(
    encoding_model: Model,
    model_name: str,
    image_path: Path,
    annotation_path: Path,
    output_folder: Path,
    target_spacing,
    context_spacing,
    width,
    height,
    labels,
    max_size: int,
):
    records = []
    wsi = WholeSlideImage(image_path)
    annotations = WholeSlideAnnotation(annotation_path, labels=labels).annotations

    for annotation in enumerate(annotations):
        if annotation.size[0] > max_size and annotation.size[1] > max_size:
            print('annotation to big...', annotation.label.name)
            continue

        record = {}

        x,y = np.array(annotation.centroid)
        target_patch = wsi.get_patch(x,y, width, height, target_spacing)
        context_patch = wsi.get_patch(x,y, width, height, context_spacing)
        encoding = encoding_model.predict([np.array([target_patch]), np.array([context_patch])])[0]

        record['x'] = x
        record['y'] = y
        record['width'] = width
        record['height'] = height
        record['model_name'] = model_name
        record['annotation'] = annotation
        record['annotation_path'] = annotation_path
        record['max_size'] = max_size
        record['image_path'] = image_path
        record['label'] = annotation.label.name
        record['target_spacing'] = target_spacing
        record['context_spacing'] = context_spacing
        record['encoding'] = encoding
   
        records.append(record)

    wsi.close()
    output_path = output_folder / Path(annotation_path.stem + f'_encodings_{model_name}.npy')
    np.save(output_path, records)

