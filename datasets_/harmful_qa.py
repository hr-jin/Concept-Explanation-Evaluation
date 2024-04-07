from .base import AbstractDataset
import os
import pandas as pd
import numpy as np

class HarmFulQADataset(AbstractDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_dataset()
            
    @classmethod
    def code(cls):
        return 'HarmfulQA'

    def load_dataset(self):
        data_dir = self.cfg['data_dir']
        
        # df = pd.read_csv(os.path.join(data_dir, 'advbench+harmfulqa.csv'))
        # df = pd.read_csv(os.path.join(data_dir, 'HarmfulQA.csv'))
        
        
        # df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset_shuffled.csv'))
        
        if self.cfg['tcav_dataset'] == 'DD':
            df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset.csv'))
            topic = 'Disinformation and deception'
            # pos_data = [' '.join([i, '\nThe amount of information in the abve sentence containing', '"' + topic + '"', 'is'])  for i in df[df['category'] == topic]['forbidden_prompt'].tolist()]
            # neg_data = [' '.join([i, '\nThe amount of information in the abve sentence containing', '"' + topic + '"', 'is'])  for i in df[df['category'] != topic]['forbidden_prompt'].tolist()]
            pos_data = df[df['category'] == topic]['forbidden_prompt'].tolist()
            neg_data = df[df['category'] != topic]['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'HHD':
            df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset.csv'))
            topic = 'Hate, harassment and discrimination'
            pos_data = df[df['category'] == 'Hate, harassment and discrimination']['forbidden_prompt'].tolist()
            neg_data = df[df['category'] != 'Hate, harassment and discrimination']['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'IGS':
            df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset.csv'))
            topic = 'Illegal goods and services'
            pos_data = df[df['category'] == 'Illegal goods and services']['forbidden_prompt'].tolist()
            neg_data = df[df['category'] != 'Illegal goods and services']['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'NVC':
            df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset.csv'))
            topic = 'Non-violent crimes'
            pos_data = df[df['category'] == 'Non-violent crimes']['forbidden_prompt'].tolist()
            neg_data = df[df['category'] != 'Non-violent crimes']['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'SC':
            df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset.csv'))
            topic = 'Sexual content'
            pos_data = df[df['category'] == 'Sexual content']['forbidden_prompt'].tolist()
            neg_data = df[df['category'] != 'Sexual content']['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'V':
            df = pd.read_csv(os.path.join(data_dir, 'strongreject_dataset.csv'))
            topic = 'Violence'
            pos_data = df[df['category'] == 'Violence']['forbidden_prompt'].tolist()
            neg_data = df[df['category'] != 'Violence']['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'ADV+HQA':
            df = pd.read_csv(os.path.join(data_dir, 'advbench+harmfulqa.csv'))
            topic = 'harmful information'
            pos_data = df['question_1'].tolist()
            neg_data = df['question_2'].tolist()   
        elif self.cfg['tcav_dataset'] == 'IA':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Illegal Activity'
            pos_data = df[df['content_policy_name'] == 'Illegal Activity']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Illegal Activity']['forbidden_prompt'].tolist()
        elif self.cfg['tcav_dataset'] == 'HS':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Hate Speech'
            pos_data = df[df['content_policy_name'] == 'Hate Speech']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Hate Speech']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'M':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Malware'
            pos_data = df[df['content_policy_name'] == 'Malware']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Malware']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'PH':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Physical Harm'
            pos_data = df[df['content_policy_name'] == 'Physical Harm']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Physical Harm']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'EH':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Economic Harm'
            pos_data = df[df['content_policy_name'] == 'Economic Harm']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Economic Harm']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'F':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Fraud'
            pos_data = df[df['content_policy_name'] == 'Fraud']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Fraud']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'P':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Pornography'
            pos_data = df[df['content_policy_name'] == 'Pornography']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Pornography']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'PL':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Political Lobbying'
            pos_data = df[df['content_policy_name'] == 'Political Lobbying']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Political Lobbying']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'PV':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Privacy Violence'
            pos_data = df[df['content_policy_name'] == 'Privacy Violence']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Privacy Violence']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'LO':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Legal Opinion'
            pos_data = df[df['content_policy_name'] == 'Legal Opinion']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Legal Opinion']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'FA':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Financial Advice'
            pos_data = df[df['content_policy_name'] == 'Financial Advice']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Financial Advice']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'HC':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Health Consultation'
            pos_data = df[df['content_policy_name'] == 'Health Consultation']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Health Consultation']['forbidden_prompt'].tolist() 
        elif self.cfg['tcav_dataset'] == 'GD':
            df = pd.read_csv(os.path.join(data_dir, 'DAN_dataset.csv'))
            topic = 'Gov Decision'
            pos_data = df[df['content_policy_name'] == 'Gov Decision']['forbidden_prompt'].tolist()
            neg_data = df[df['content_policy_name'] != 'Gov Decision']['forbidden_prompt'].tolist() 
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_HHD_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_DD_pos.csv'))
        # df = pd.read_csv(os.path.join(data_dir, 'strongreject_IGS_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_NVC_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_SC_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_V_pos.csv'))
        # pos_data = df['question_1'].tolist()
        
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_HHD_neg.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_DD_neg.csv'))
        # df = pd.read_csv(os.path.join(data_dir, 'strongreject_IGS_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_NVC_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_SC_pos.csv'))
        # # df = pd.read_csv(os.path.join(data_dir, 'strongreject_V_pos.csv'))
        # neg_data = df['question_2'].tolist()   
        pos_labels = np.ones((len(pos_data), ))  
        neg_labels = np.zeros((len(neg_data),))      
        self.data = (pos_data, neg_data, pos_labels, neg_labels)


