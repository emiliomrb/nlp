#Tis is my firs experimient with Flair library. It's a entity extractor from news articles. I took news from The Gusrdian
#and put the url in the url param.

# I use BeautifulSoup for scrap de web siite and extract the text





from urllib.request import urlopen
import pandas as pd
from flair.data import Sentence, Token
from flair.embeddings import StackedEmbeddings, WordEmbeddings, FlairEmbeddings, CharacterEmbeddings, BytePairEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.nn import Model
import requests
from bs4 import BeautifulSoup

from segtok.segmenter import split_single
def entities_extractor(url):


  res = requests.get(url)
  html_page = res.content

  soup = BeautifulSoup(html_page, 'html.parser')
  text = soup.find_all(text=True)
  set([t.parent.name for t in text])

  output = ''
  blacklist = [
      '[document]',
       'a',
       'article',
       'aside',
       'body',
       'button',
       'clippath',
       'defs',
       'div',
       'figcaption',
       'figure',
       'footer',
       'form',
       'g',
       'h1',
       'h2',
       'head',
       'header',
       'html',
       'label',
       'li',
       'link',
       'meta',
       'nav',
       'noscript',
       'picture',
       'script',
       'section',
       'span',
       'strong',
       'style',
       'svg',
       'time',
       'title',
       'ul',
      # there may be more elements you don't want, such as "style", etc.
  ]

  for t in text:
      if t.parent.name not in blacklist:
          output += '{} '.format(t)

  sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(output)]
  
  tagger = SequenceTagger.load('ner')
  tagger.predict(sentences)
  
  li = []
  for i in sentences:
      for entity in i.get_spans('ner'):
          li.append(entity.to_dict())

  df = pd.DataFrame(li)
  df= pd.crosstab(df.text, df.type)
  return df
