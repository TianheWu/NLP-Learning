import spacy


nlp = spacy.load("en_core_web_sm")
doc = nlp("WuTianhe is going to fuck LiuWenxing")
for chunk in doc:
    print('{} - {}'.format(chunk, chunk.pos_))