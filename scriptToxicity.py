
import torch
from rewardlm.core.GenerativeModel import GenerativeModel
from rewardlm.ToxicityMeter import ToxicityMeter
from rewardlm.data.data_utils import get_real_toxicity_prompts
from rewardlm.utils import load_config

from argparse import ArgumentParser
from huggingface_hub import login
import datetime
now = datetime.datetime.now()   # getting current date for log

def main(config_name: str):
    print(now)
    
    print(f'[-] Loading {config_name} config')
    config = load_config(name=config_name)

    generator_manager = GenerativeModel(
        config['model_id'],
        load_from_peft=config['load_from_peft'],
        generation_config=config['generation']['generation_config'],
        accelerator_kwargs = {
            'cpu': False if torch.cuda.is_available() else True,
        },
    )

    # leaving the default reward manager
    toxicity_meter = ToxicityMeter(generator_manager)

    custom_prompt = (config['generation']['custom_prompt']['user_name'] + 
                    ' "{prompt}".\n' + 
                    config['generation']['custom_prompt']['bot_name'] + ' '
                    )

    df = get_real_toxicity_prompts()
    toxicity_df = toxicity_meter.measure_toxicity(
        text_prompt=df if not config['data']['subset'] else df[:config['data']['subset_size']],
        custom_prompt=custom_prompt,
        batch_size=config['inference_batch_size'],
        print_response=config['debug'],     # print responses
    )
    
    # save csv in tmp folder
    fldr = './result analysis/new_prompts'
    toxicity_df.to_csv(fldr + f'/measured_tox_{config["model_id"].split("/")[-1]}.csv')




if __name__ == "__main__":
    parser = ArgumentParser(description='Get config file.')
    parser.add_argument('-c', '--config', required=True, help='Config name (without the .yaml). Files are stored in PROJ_PATH/configs/*.yaml')

    args = parser.parse_args()
    config_name = args.config

    # in case models are private (no model storing in this script)
    credentials = load_config(path = './', name = 'credentials')
    login(token = credentials['huggingface_hub'])

    main(config_name = config_name)