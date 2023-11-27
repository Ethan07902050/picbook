from abc import ABC
import json
import logging
import os
import zipfile

import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import load_from_disk
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersQuestionHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersQuestionHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        # Read model serialize/pt file
        self.model = DPRQuestionEncoder.from_pretrained(model_dir)
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        zip_path = os.path.join(model_dir, 'book_dataset.zip')
        dataset_path = os.path.join(model_dir, 'book_dataset')
        index_path = os.path.join(model_dir, 'book_index.faiss')

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        self.dataset = load_from_disk(dataset_path)
        self.dataset.load_faiss_index('embeddings', index_path)

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        logger.info(f'data: {data[0]}')
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        inputs = self.tokenizer(sentences, return_tensors="pt")
        return inputs

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'].to(self.device))
        query_embedding = outputs[0][0].cpu().numpy()
        scores, examples = self.dataset.get_nearest_examples('embeddings', query_embedding, k=1)
        _id = examples['id'][0]
        logger.info(f"predicted id {_id}")
        return [{'id': _id}]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersQuestionHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e