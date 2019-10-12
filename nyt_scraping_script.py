import requests
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import config
import logging
import argparse
import json
import pickle
import re
import time
import random



# Set logging parameters
# Set logging parameters
logging.basicConfig(level=logging.INFO, filename='nyt_scraping_script.log', filemode='w+', 
                    format='%(asctime)-15s:  %(levelname)s - %(message)s')


def get_nyt_article(soup):
    """
    get_nyt_article(soup):
    Given beautiful soup for a given NYT article,
    pull out all text paragraphs, clean and strip,
    and return as a single string.
    Params:
        soup:   Beautiful Soup object containing the
                'soupified' article
    Returns:
        String containing the article's text
    """
    # all article text is contained in paragraphgs of this class
    article_list = soup.find_all('p', class_='css-exrw3m evys1bk0')

    # discard embedded NYT notices, e.g., '[Sign up....]'
    pat = r'\[[a-zA-Z].*\]'
    regex_match = re.compile(pat)
    text_list = []
    for item in article_list:
        text = regex_match.sub('', item.text)
        text.strip()
        text_list.append(text)
    
    return ''.join(text_list)


def get_nyt_url_dicts(data):
    nyt_url_dicts = []
    for doc in data['response']['docs']:
        section_name = doc['section_name']
        if section_name == 'U.S.' or section_name == 'World':
            # make sure the url doesn't link to a video,
            # else we won't get any text - so skip
            # TO DO: make a better determination (some flag?)!
            url = doc['web_url']
            vid = r'/video'
            if url.find(vid) > -1:
                continue
            
            base_dt = doc['pub_date']
            dt = base_dt[0:base_dt.find('T')]
            dict_entry = dict(date=dt,
                             section=section_name,
                             headline=doc['headline']['main'],
                             url=url)
            nyt_url_dicts.append(dict_entry)
    
    return nyt_url_dicts


def get_nyt_month_archive(year, month):
    base_search_url = 'https://api.nytimes.com/svc/archive/v1'
    url = f'{base_search_url}/{year}/{month}.json?api-key={config.nyt_api_key}'
    
    try:
        response = requests.get(url)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        logging.exception(f'HTTP error occurred: {http_err}')  # Python 3.6
    except Exception as err:
        logging.exception(f'Other error occurred: {err}')  # Python 3.6
    else:
        logging.info(f'fetched archive for {month}/{year}')
        return json.loads(response.text)



def main():
    # parser = argparse.ArgumentParser(description='Process some args.')
    # parser.add_argument('month',
    #                     metavar='M',
    #                     type=str, help='month to process')

    # args = parser.parse_args()
    # print(args)
    # print(args.month)

    # logging.info(f'month arg: {args.month}')

    # set up sequence of sleep time to be randomized
    sleep_sequence = [x/10 for x in range(8, 14)]


    year = '2019'
    month = '10'

    logging.info(f'API call for {month} {year}')

    json_dict = get_nyt_month_archive(year, month)

    # save to file
    arch_file_name = f'nyt_{month}_{year}_archive.json'
    article_file_name = 'nyt_test.json'
    pickle_file_name = 'nyt_test_articles.pkl'

    with open(arch_file_name, 'w') as arch_file, open(article_file_name, 'w+') as article_file, open(pickle_file_name, 'wb') as pickle_file:
        arch_file.write(json.dumps(json_dict))

        # get a dict of article urls of interest:
        article_url_dicts = get_nyt_url_dicts(json_dict)

        logging.info(f'Fetched {len(article_url_dicts)} article URL entries')
        logging.info('Fetching each article...')

        article_collection = []
        for doc in article_url_dicts:
            # get url:
            url = doc['url']
            dt = doc['date']
            section = doc['section']
            headline = doc['headline']

            try:
                response = requests.get(url)
                # If the response was successful, no Exception will be raised
                response.raise_for_status()
            except HTTPError as http_err:
                logging.exception(f'HTTP error occurred: {http_err} - url: {url}')  # Python 3.6
                continue
            except Exception as err:
                print(f'Other error occurred: {err} - url: {url}')  # Python 3.6
                continue
            else:
                # success - do the text extraction!
                soup = BeautifulSoup(response.content, 'html.parser')
                article = get_nyt_article(soup)

                # log it, then write it!
                logging.info(f"date: {dt} | section: {section} | url: {url} | headline: {headline} | text: {article[0:20]}")

                # create a data structure with our text
                text_entry = dict(paper='NYT',
                                  date=dt,
                                  section=section,
                                  url=url,
                                  headline=headline,
                                  text=article)
                article_file.write(json.dumps(text_entry))
                article_file.write('\n')
                
                article_collection.append(text_entry)

                # now sleep a random period of time
                time.sleep(random.choice(sleep_sequence))
 

        pickle.dump(article_collection, pickle_file)





if __name__ == '__main__':
    #
    # test our script with constrained API call and
    main()
