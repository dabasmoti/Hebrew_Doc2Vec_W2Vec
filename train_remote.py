

import os
import sys
import time
import datetime
import argparse
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import hebrew_tokenizer as ht
import pandas as pd








def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',help= 'path to txt files')
    parser.add_argument('--file_type',help= 'path to txt files',default='.txt')
    parser.add_argument('--csv',help= 'path to csv files')
    parser.add_argument('--csv_encoding',help= 'encoding for the csv file default utf-8-sig',default='utf-8-sig')
    parser.add_argument('--sort_col',help= 'argument for sort by',default='brand')
    parser.add_argument('--tag_col',help= 'argument for tag column by',default='article_id')
    parser.add_argument('--sort_date_col',help= 'column to sort by',default='publish_time')
    parser.add_argument('--out_put',help= 'path to save pickle file')
    parser.add_argument('--pickle_file',help= 'path to load pickle file',default=None)
    parser.add_argument('--dm',help= 'path to save pickle file',type=int,default=0)
    parser.add_argument('--max_files',help= 'Number of files to process',type=int,default=None)
    parser.add_argument('--window',help= 'window words',type=int,default=5)
    parser.add_argument('--remove_stp',help= 'yes to remove stopwords ',default='no')
    parser.add_argument('--epoches',help= 'Number of epoches to process',type=int,default=15)
    parser.add_argument('--vec_size',help= 'vector size',type=int,default=300)
    parser.add_argument('--min_count',help= 'min count ',type=int,default=5)
    parser.add_argument('--hs',help= 'Hierarchical softmax ',type=int,default=0)
    parser.add_argument('--workers',help= 'Number of cpus  ',type=int,default=os.cpu_count())
    parser.add_argument('--dbow_words',help= 'dbow_words  ',type=int,default=1)
    parser.add_argument('--negative',help= 'negative words  ',type=int,default=5)
    parser.add_argument('--sample',help= 'sample   ',type=int,default=0)
    parser.add_argument('--save',help= 'path save ',default='/root/moti/doc2vec/models')

    args = parser.parse_args()
    return args


def stop_words(): return  set(
    """
    של
    את
    על
    לא
    הוא
    עם
    זה
    גם
    כי
    היא
    אבל
    כל
    או
    היה
    אני
    יותר
    מה
    אם
    ב
יש
הם
בין
כך
רק
כדי
כמו
אחד
עד
ל
שלו
אותו
אך
לפני
לו
שלא
לי
אמר
עוד
זאת
היתה
הזה
אחרי
כבר
כמה
זו
לאחר
היו
היום
להיות
שהוא
ולא
אל
מאוד
אלא
מי
בו
בכל
ה
שני
ביותר
שם
שנים
בית
שלי
בן
אז
אנחנו
אחת
אותה
אחר
נגד
שלה
אפשר
אף
לה
הרבה
בני
מ
באופן
שבו
לפי
דבר
להם
""".split()
)


def get_tokens(p,remove_stp=False):
    with open(p,'r',encoding='utf-8') as f:
        if remove_stp:
            return [token for grp, token, token_num, (start_index, end_index) in ht.tokenize(f.read()) if grp == 'HEB' and token not in stop_words()]  
        else:
            return [token for grp, token, token_num, (start_index, end_index) in ht.tokenize(f.read()) if grp == 'HEB' ] 



class DataframeCorpus(object):
    def __init__(self, source_df, path_col, tag_col):
        self.source_df = source_df
        self.path_col = path_col
        self.tag_col = tag_col
        self.remove_stp = True if args.remove_stp == 'yes' else False
        

    def __iter__(self):
        for i, row in self.source_df.iterrows():
            
            yield TaggedDocument(words=get_tokens(row[self.path_col],remove_stp=self.remove_stp), 
                                 tags=[row[self.tag_col]])



def train_d2vec(_df,args):
    model = (Doc2Vec(vector_size=args.vec_size,epochs=args.epoches, min_count=args.min_count,
                     dm=args.dm,negative=args.negative,workers=args.workers,dbow_words=args.dbow_words,
                     hs=args.hs,sample=args.sample ))
    
    corpus = DataframeCorpus(_df,'path',args.tag_col)
    logging.info('.. Build vocabulary')
    model.build_vocab(corpus)
    print('total corpus count',model.corpus_count)
    print('mopdel.epochos = ',model.epochs)
    logging.info('.. Train model')
    model.train(corpus,total_examples=model.corpus_count,epochs=model.epochs)

    return model


def name_it(args):
    fold_name = args.save
    if args.dm == 1:
        fold_name += '/dm_'
    else: 
        fold_name += '/dbow_W'
    fold_name += str(args.window) + '_EP' + str(args.epoches)+ '_' + args.remove_stp +'_remove_stp' + '_' + str(datetime.datetime.today())[0:16].replace(' ','_') +'/'
    model_name = ''
    if args.dm == 1:
        model_name += 'dm_w'
    else:
        model_name += 'dbow_w'
    model_name += str(args.window) + '_EP' + str(args.epoches) + '_' + args.remove_stp +'_remove_stp' 
    return fold_name, model_name


def load_df(args):
    """read csv file to dataframe by passed arguments """
    
    logging.info('.. Load DataFrame')
    df = (pd.read_csv(args.csv,dtype={args.tag_col:str},
                      parse_dates=[args.sort_date_col],
                      usecols=[args.tag_col,args.sort_col,args.sort_date_col],
                      encoding = args.csv_encoding,
                      nrows = args.max_files))
    #Sort by Arguments
    df = df.sort_values(by=[args.sort_col,args.sort_date_col], ascending=[False,True])
    #Add path column
    df['path'] = args.data_path + '/' + df[args.tag_col] + args.file_type
    
    return df

    

def main(args):
    
    fold_name, model_name = name_it(args)
    print('folder name', fold_name)
    print('model name', model_name)
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)    
    else:
        os.makedirs(fold_name+str(datetime.datetime.today())[0:18])
    
    #logging.basicConfig(filename=fold_name +'/logs.txt' ,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(fold_name + "/logs.log"),
        logging.StreamHandler()
    ]
)

    

    path = args.data_path
    print(path)
    df = load_df(args)
    
    print('Start training..')

    time_0 = time.time()
    model_path = fold_name+model_name
    model = train_d2vec(df,args)
    print('Done training in {} minutes'.format((round(time.time()-time_0)/60)))
    model.save(model_path)
    with open(fold_name+'/model_desc.txt','w') as desc:
        desc.write(str(model)+'\n')
        for line in str(args).split(','):            
            desc.write(line+'\n')
        desc.close()
    
        
     

if __name__ == '__main__':
    args = sys.argv[1:]       
    args = parse_args(args)

    main(args)

