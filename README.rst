Compressing T5 models for summarization
=======================================

My quest to create a summarizer model that is as small as possible,
yet produce good results, is still going on.

I've stumbled on the https://github.com/JulesBelveze/bert-squeeze project which
is a pretty cool project that applies different strategies to compress models
and wanted to try it on a longt5 model.

Unfortunately, the project does not support seq2seq models yet. But Jules is
a very nice guy and is willing to take my contribution and helped me understand
a few things around models.

He pointed me to a research paper about the "Shrink and Fine-tune" strategy,
(see https://arxiv.org/pdf/2010.13002.pdf) which consist of pruning some decoder
layers from the original model and
then fine-tuning it again with the data that was used to train the original model.

So it's not a distillation per se, it's a fine-tuning on a model that has some
of the initial layers removed.

I gave up on the idea of training longt5 models on my Apple M1 because the 32GiB
memory that is shared between the GPU and the CPU dies pretty quickly given
the size of the data. longt5 accepts 16k tokens instead of 512, which blows my
memory. PyTorch's GPU backend is also three times slower than the CPU for me,
which I realized after some experiments.

To summarize, training models on an Apple laptop is not great.
This is why I've ordered 2 RTX-4090 - I can't wait to play with bigger models.


Shrinking
#########


Until then, I tried the SFT strategy on a smaller model. I took Jule's tldr model
https://huggingface.co/JulesBelveze/t5-small-headline-generator and applied the shrinking.

I tried to shrink just the decoder layers and also both the encoder and decoder.
The former reduces the size from 60.6M params to 47.9M and the latter to 38.5M.

This is the gist of the code used to shrink layers using transformers:

..code-block:: python

  def shrink_layer(model, layer_name, new_size=None):
    if layer_name == "encoder":
        config_name = "num_layers"
    else:
        config_name = f"num_{layer_name}_layers"

    current_size = getattr(model.config, config_name)

    if new_size is None:
        new_size = int(current_size / 2)

    if current_size != new_size:
        layers_to_remove = [i for i in range(1, new_size * 2, 2)]

        for i in reversed(layers_to_remove):
            del getattr(model, layer_name).block[i]

        setattr(model.config, config_name, new_size)


  def load_and_shrink_t5_model(
      model_name, num_decoder_layers=None, num_encoder_layers=None
  ):
      model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
      print("Shrinking model...")
      shrink_layer(model, "decoder", num_decoder_layers)
      shrink_layer(model, "encoder", num_encoder_layers)
      return model


  # creating one with 6 encoder and 3 decoder layers
  load_and_shrink_t5_model("JulesBelveze/t5-small-headline-generator", 3, 6)

  # creating one with 3 encoders and 3 decoder layers
  load_and_shrink_t5_model("JulesBelveze/t5-small-headline-generator", 3, 3)


Training
########

TODO


Once saved and quantized, the smallest model is down to 50MiB (as opposed to 250MiB) !

Using the demo script, from Jules' model card, I get those summaries:

- Original: US FCC commissioner asks Apple and Google to remove TikTok from app stores
- 50% decoder layers: Apple and Google to remove TikTok from their app stores
- 50% encoder and decoder layers: China’s TikTok says it can harvest data from U.S. citizens

Evaluation
##########

TODO

