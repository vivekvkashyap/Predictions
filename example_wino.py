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

num_choices=2
win0={'sentence':'Kevin had to have well manicured nails for work, but not Joseph, because _ worked at a bank.','option1': 'Joseph','option2': 'Kevin'}
win1={'sentence':'I went into the mine looking for gold but all I found was topaz and quartz.  I found on the _ because it was almost clear.','option1': 'topaz','option2': 'quartz'}
win2={'sentence':'Joseph immediately went to bank before the bakery because the _ had a substantial supply of what he wanted.','option1': 'bakery','option2': 'bank'}
win3={'sentence':'Maria was a much better lawyer than Sarah so _ always got the easier cases.','option1': 'Sarah','option2': 'Maria'}
win4={'sentence':'California is home to Kayla , but Elena calls Illinois home, so _ has warm winters.','option1': 'Elena','option2': 'Kayla'}
win5={'sentence':'Nelson let Robert burn down  the trash as they were to take turns and _ less recently.','option1': 'Nelson','option2': 'Robert'}
win_total=[win0,win1,win2,win3,win4,win5]
win={}
win['sentence']=[i['sentence'] for i in win_total]
win['option1']=[i['option1'] for i in win_total]
win['option2']=[i['option2'] for i in win_total]
win_dataset = Dataset.from_dict(win)

def preprocess(example):
    example['first_sentence']=[example['sentence']]*num_choices
    example['second_sentence']=[example[f'option{i}'] for i in [1,2]]
    return example

wwin_dataset=win_dataset.map(preprocess)

len_test_dataset=6

tokenizer=GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B',pad_token='<|endoftext|>')

remove_col=wwin_dataset.column_names

def tokenize(examples):
    tokenized_examples=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=512,return_tensors='jax')
    return tokenized_examples

test_dataset=wwin_dataset.map(tokenize)

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

from  model_file  import FlaxGPTNeoForMultipleChoice

model = FlaxGPTNeoForMultipleChoice.from_pretrained('Vivek/gptneo_winogrande',input_shape=(1,num_choices,1))

restored_output=[]
b=jnp.array(test_dataset['input_ids'])
c=jnp.array(test_dataset['attention_mask'])
outputs=model(b,c)
print(outputs)
final_output=jnp.argmax(outputs,axis=-1)
restored_output.append(final_output)

finall=pd.DataFrame({'predictions':restored_output})
finall.to_csv('./robustness_wino.csv')