import json
import re
from collections import Counter
import sys
import jieba

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        self.use_jieba = args.use_jieba

        if self.dataset_name == 'mimic_abn':
            self.clean_report = self.clean_report_mimic_cxr
            self.token2idx, self.idx2token = self.create_vocabulary()

        elif self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
            self.token2idx, self.idx2token = self.create_vocabulary()

        elif self.dataset_name == 'mimic_cxr':
            self.clean_report = self.clean_report_mimic_cxr
            self.token2idx, self.idx2token = self.create_vocabulary()

        elif self.dataset_name == 'zju2':
            self.clean_report = self.clean_report_zju2
            self.token2idx, self.idx2token = self.create_vocabulary()
          
    def create_vocabulary(self):
        self.ann = json.loads(open(self.ann_path, 'r').read())

        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report'])#.split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token


    def create_vocabulary_mimic_abn(self):
        def load_captions(data_dir, splits):
            with open(data_dir) as file:
                all_caps = json.load(file)

            data = []
            for split in splits:
                with open(data_dir+split+'_split.json') as file:
                    img_names = json.load(file)
                    data += [','.join(all_caps[i]['sents']) for i in img_names]

            return data

        captions = load_captions(self.ann_path, ['train'])
        total_tokens = []
        for cap in captions:
            tokens = self.clean_report(cap)#.split()
            for token in tokens:
                total_tokens.append(token)
        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        report = report.split()#将report按单词划分
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        report = report.split()#将report按单词划分
        return report

    def clean_report_zju2(self, report):
        report_cleaner = lambda t: t.replace('1:', '').replace('1 ', '').replace('2:', '').replace('2 ', '')\
            .replace('FFA提示：', '').replace('FFA提示', '').replace('FFA示：', '').replace('FFA示', '')\
            .replace('FFA', '').replace('ＦＦＡ示', '').replace('FA', '').replace('造影示：', '').replace('造影示', '')  \
            .replace(',','，').replace('.','。').replace(' ','') \
            .replace('左眼', '').replace('右眼', '').replace('IRMA', '微血管异常') \
            .replace('视乳头', '视盘').replace('玻血', '玻璃体积血').replace('晚期', '后期').replace('光凝斑', '激光斑')  \
            .replace('少许', '少量').replace('扭曲', '迂曲').replace('玻璃体浑浊', '屈光介质浑浊').replace('视盘周围', '盘沿') 
        if self.use_jieba == True:
            jieba.load_userdict('/data2/liuxiaocong/oph/RepGen-main/data/zju2/jieba_dic.txt')
            report = jieba.lcut(str(report_cleaner(report))) #按词划分
        else:
            report = list(report_cleaner(report))#按字划分
        return report

    def clean_report_mimic_abn(self, report):
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report)#.split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
