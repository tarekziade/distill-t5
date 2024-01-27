
.PHONY: quantize
quantize: bin/python convert.py --quantize yes --model_id ./t5-small-headline-generator-sft-3-3/ --task text2text-generation-with-past

