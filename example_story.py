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

story0={'Full_sentence':'Kate was having a school fundraiser for Krispy Kreme Doughnuts.Kate was selling the doughnuts for band uniforms.In addition, the top seller would get a monetary prize.Kate worked hard over the next two weeks selling doughnuts.','choice1': 'Kate could not win the monetary prize.','choice2': 'Kate won one hundred dollars.'}
story1={'Full_sentence':'Drew order a small ice cream cone at the Drive Thru.He drove to the front window to wait for his order.The cashier handed the ice cream with one of her hands.Drew took the ice cream cone and did not turn it upside down.','choice1': 'Drew made a mess.','choice2': 'Drew kept his car clean this way.'}
story2={'Full_sentence':"Johnny wanted a change from his big city life in Boston.He decided he would be happier in America's Heartland.Finally the day had arrived, Johnny moved to Idaho.But he did  not like the small town he thought big city was better",'choice1': 'Johnny again moved back to big citys life','choice2': 'Johnny was happy to be in a small town.'}
story3={'Full_sentence':'After Igor woke up he began to get ready for work.He turned on the shower and waited for cold water.The water remained hot.He debated between taking a hot shower and not showering.','choice1': 'Igor took a hot shower','choice2': 'Igor took a cold shower.'}
story_total=[story0,story1,story2,story3]
story={}
story['Full_sentence']=[i['Full_sentence'] for i in story_total]
story['choice1']=[i['choice1'] for i in story_total]
story['choice2']=[i['choice2'] for i in story_total]
story_dataset = Dataset.from_dict(story)


def preprocess(example):
    example['first_sentence']=[example['Full_sentence']]*num_choices
    example['second_sentence']=[example[f'choice{i}'] for i in [1,2]]
    return example

sstory_dataset=story_dataset.map(preprocess)

len_test_dataset=4

tokenizer=GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B',pad_token='<|endoftext|>')

remove_col=sstory_dataset.column_names

def tokenize(examples):
    tokenized_examples=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=512,return_tensors='jax')
    return tokenized_examples

test_dataset=sstory_dataset.map(tokenize)

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

model = FlaxGPTNeoForMultipleChoice.from_pretrained('Vivek/gptneo_storycloze',input_shape=(1,num_choices,1))

restored_output=[]
rng, input_rng = jax.random.split(rng)
b=jnp.array(test_dataset['input_ids'])
c=jnp.array(test_dataset['attention_mask'])
outputs=model(b,c)
print(outputs)
final_output=jnp.argmax(outputs,axis=-1)
restored_output.append(final_output)

finall=pd.DataFrame({'predictions':restored_output})
finall.to_csv('./robustness_storycloze.csv')