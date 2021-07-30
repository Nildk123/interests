import argparse
import yaml
import os
import time
from src.result.model_output_generation import make_evaluation 

with open("cnfg.yml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
		
# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--num_class', type=int, help="The number of classes for test or validation", nargs = '?', const = 1, default = 3)
parser.add_argument('-i', '--input_source', type=int, help="Input foom flat file or DB, 1 for DB", nargs = '?', const = 1, default = 0)
parser.add_argument('-t', '--is_train', type=int, help="use for train or test, 1 for train", nargs = '?', const = 1, default = 0)
parser.add_argument('-f', '--input_file_name', type=str, help="get file name if flat file source is defined", nargs = '?', const = 1, default = 'input_data.csv')
parser.add_argument('-d', '--device', type=str, help="select device (cpu, cuda) for model run", nargs = '?', choices = ['cpu', 'cuda'], default = 'cpu') 
args = parser.parse_args()
print(args)

config['file_input']['use_DB'] = args.input_source
config['model_input']['num_class'] = args.num_class
config['model_input']['is_train'] = args.is_train
config['file_input']['input_file_name'] = args.input_file_name
config['model_input']['device'] = args.device
print('Model is running for {0} classes on {1}...'.format(config['model_input']['num_class'], config['model_input']['device']))
   
if __name__ == '__main__':
    start = time.time()
    df = make_evaluation(config)
    if config['model_input']['num_class'] > 2:
        save_name = os.path.join(os.getcwd(), config['file_output']['output_dir'], config['file_output']['file_name'][1])
    else:
        save_name = os.path.join(os.getcwd(), config['file_output']['output_dir'], config['file_output']['file_name'][0])
    df.to_csv(save_name, index = False)
    done = time.time()
    elapsed = done - start
    print('time taken to complete the module is {:.2f} seconds'.format(round(elapsed, 2)))