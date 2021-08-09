import jax
print(jax.local_device_count())
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.training.common_utils import get_metrics,onehot,shard,shard_prng_key

from transformers import  GPTNeoConfig
from transformers.models.gpt_neo.modeling_flax_gpt_neo import FlaxGPTNeoPreTrainedModel
from transformers import GPT2Tokenizer

from datasets import load_dataset,Dataset
import pandas as pd

num_choices=4
dataset = load_dataset("hellaswag")

hell2={'ctx_a':'We are instructed and shown how to use the sharpening steel. We are instructed and shown how to use a mug to sharpen a knife.','ctx_b':'we','endings':['are shown the knife cutting carrots','see the shuttlecock in the image.','see the lady in red again.','are shown the polish and prep display.']}
hell3={'ctx_a': 'The first jump is poor, and he knocks off the bar during the jump. He clears the bar on the second jump.','ctx_b': 'the third jump','endings': ['is high enough and he sits on the bar.','is extremely easy, but does not last.','is the most impressive, with the athlete clearing the bar again.','is better, he falls off and almost regains his balance.']}
hell4={'ctx_a': 'Several different men are seen trying to snow board but, falling down over and over.','ctx_b': 'many different people','endings': ['are shown falling down while in the air and occasionally throwing themselves off.','are shown going around on skis showing the runs they have taken and skiing.','are in the background watching and doing different things in the snow.','watch on the sides as the camera flies around the city and continues going on with their walk and end by curling onto the ground.']}
hell_total=[hell2,hell3,hell4]
hell={}
hell['ctx_a']=[i['ctx_a'] for i in hell_total]
hell['ctx_b']=[i['ctx_b'] for i in hell_total]
hell['endings']=[i['endings'] for i in hell_total]

def preprocess(example):
    example['first_sentence']=[example['ctx_a']]*4
    example['second_sentence']=[example['ctx_b']+' '+example['endings'][i] for i in range(4)]
    return example

hell_dataset = Dataset.from_dict(hell)
hhell_dataset=hell_dataset.map(preprocess)

len_test_dataset=4

tokenizer=GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B',pad_token='<|endoftext|>')

remove_col=hhell_dataset.column_names

def tokenize(examples):
    tokenized_examples=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=256,return_tensors='jax')
    return tokenized_examples

test_dataset=hhell_dataset.map(tokenize)

test_dataset=test_dataset.remove_columns(remove_col)
list1=[]

def glue_test_data_loader(rng,dataset,batch_size):
  steps_per_epoch=len_test_dataset//batch_size
  perms=jax.random.permutation(rng,len_test_dataset)
  perms=perms[:steps_per_epoch*batch_size]
  perms=perms.reshape((steps_per_epoch,batch_size))
  for perm in perms:
    list1.append(perm)
    batch=dataset[perm]
    #print(jnp.array(batch['label']))
    batch={k:jnp.array(v) for k,v in batch.items()}
    #batch=shard(batch)
    yield batch

seed=0
rng=jax.random.PRNGKey(seed)
dropout_rngs=jax.random.split(rng,jax.local_device_count())

input_id=jnp.array(test_dataset['input_ids'])
att_mask=jnp.array(test_dataset['attention_mask'])

total_batch_size=1

from  model_file  import FlaxGPTNeoForMultipleChoice

model = FlaxGPTNeoForMultipleChoice.from_pretrained('Vivek/gptneo_hellaswag',input_shape=(1,num_choices,1))

restored_output=[]
rng, input_rng = jax.random.split(rng)
outputs=model(test_dataset['input_ids'],test_dataset['attention_mask'])
final_output=jnp.argmax(outputs,axis=-1)
restored_output.append(final_output)

finall=pd.DataFrame({'predictions':restored_output,'probability':outputs})
finall.to_csv('./robustness_hellaswag.csv')