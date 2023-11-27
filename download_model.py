from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

save_dir = './dpr_model'
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)