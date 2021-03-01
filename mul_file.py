# def get_data(path, binary):
#     files = os.listdir(path)
#     sentences = []
#     new_sentences = []
#     labels = [binary for i in range(700)]
#
#     for file in files:
#         position = path + '/' + file
#         with open(position, 'r', encoding='unicode_escape') as f:
#             data = f.read()
#             sentences.append(data)
#
#     for s in sentences:
#         s = s.replace('\n', ' ')
#         s = s.replace('.', ' ')
#         s = s.replace('!', ' ')
#         s = s.replace('-', ' ')
#         s = s.replace('/', ' ')
#         s = s.replace('_', ' ')
#         s = s.replace('#', ' ')
#         s = s.replace('$', ' ')
#         s = s.replace(';', ' ')
#         s = s.replace('[', ' ')
#         s = s.replace(']', ' ')
#         s = s.replace('~', ' ')
#         s = s.replace('|', ' ')
#         s = s.replace('(', ' ')
#         s = s.replace(')', ' ')
#         s = s.replace('=', ' ')
#         new_sentences.append(s)
#
#     return new_sentences, labels
#
#
# path_pos = "/home/wutianhe/NLP/data/mix20_rand700_tokens_cleaned/tokens/pos"
# path_neg = "/home/wutianhe/NLP/data/mix20_rand700_tokens_cleaned/tokens/neg"
#
# sentences_pos, labels_pos = get_data(path_pos, 1)
# sentences_neg, labels_neg = get_data(path_neg, 0)
#
# sentences = sentences_pos + sentences_neg
# labels = labels_pos + labels_neg