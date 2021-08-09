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
piqa0={'goal':"the water is boiling,when it's ready ,you can",'sol1':'Pour it onto a plate','sol2':'Pour it into a mug'}
piqa1={'goal':"To be able to not hear the television better,",'sol1':'increase the volume of the television speakers to an appropriate level.','sol2':'decrease the volume of the television speakers to an appropriate level.'}
piqa2={'goal':'how to take a pill?','sol1':'place the pill on your tongue and swallow with water.','sol2':'place the pill on your tongue and swallow with lemon juice.'}
piqa3={'goal':'To become more active and focused through out the day','sol1':'meditate and concentrate on everything in the present moment.','sol2':'keep your eyes closed for better focus and then try to relax and sleep.'}
piqa4={'goal':'To make your tea without sugar,','sol1':'you can use honey instead.','sol2':'you can use jelly instead.'}
piqa5={'goal':'To send an email in googel GMAIL.','sol1':'Navigate through the requirements of an google email fill them then send to the recipient who the email was intended to.','sol2':'Sign in to Gmail account then click on the compose button, fill the requirements then press send button.'}
piqa_total=[piqa0,piqa1,piqa2,piqa3,piqa4,piqa5]
piq={}
piq['goal']=[i['goal'] for i in piqa_total]
piq['sol1']=[i['sol1'] for i in piqa_total]
piq['sol2']=[i['sol2'] for i in piqa_total]
piq_dataset = Dataset.from_dict(piq)

def preprocess(example):
    example['first_sentence']=[example['goal']]*num_choices
    example['second_sentence']=[example[f'sol{i}'] for i in [1,2]]
    return example

ppiq_dataset=piq_dataset.map(preprocess)

len_test_dataset=6

tokenizer=GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B',pad_token='<|endoftext|>')

remove_col=ppiq_dataset.column_names

def tokenize(examples):
    tokenized_examples=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=512,return_tensors='jax')
    return tokenized_examples

test_dataset=ppiq_dataset.map(tokenize)

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


total_batch_size=16

from  model_file  import FlaxGPTNeoForMultipleChoice

model = FlaxGPTNeoForMultipleChoice.from_pretrained('Vivek/gptneo_piqa',input_shape=(1,num_choices,1))

restored_output=[]
b=jnp.array(test_dataset['input_ids'])
c=jnp.array(test_dataset['attention_mask'])
outputs=model(b,c)
print(outputs)
final_output=jnp.argmax(outputs,axis=-1)
restored_output.append(final_output)

finall=pd.DataFrame({'predictions':restored_output})
finall.to_csv('./robustness_piqa.csv')