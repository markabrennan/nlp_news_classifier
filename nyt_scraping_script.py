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
import validators



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
        section_name = doc.get('section_name')
#        if section_name and (section_name == 'U.S.' or section_name == 'World'):
# let's get World articles only!
        if section_name and section_name == 'World':
            # make sure the url doesn't link to a video,
            # else we won't get any text - so skip
            # TO DO: make a better determination (some flag?)!
            url = doc.get('web_url')
            if url:
                url = doc['web_url']
                vid = r'/video'
                if url.find(vid) > -1:
                    continue
            
            base_dt = doc.get('pub_date')
            if base_dt:
                dt = base_dt[0:base_dt.find('T')]

            headline_main = doc.get('headline')
            if headline_main:
                headline = headline_main.get('main')

            dict_entry = dict(date=dt,
                             section=section_name,
                             headline=headline,
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
    except requests.exceptions.HTTPError as http_err:
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
#    month = '10'

#    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# test with one month!
    month_list = [6]

    # save to file
    arch_file_name = 'nyt_url_archives_aug.json'
    article_file_name = 'nyt_data_jun.json'

    # we will not pickle, given that it is redundant
#    pickle_file_name = 'nyt_test_articles.pkl'

    with open(arch_file_name, 'a') as arch_file, open(article_file_name, 'a') as article_file:
        for month in month_list:
            logging.info(f'Calling API for month: {month}')
            json_dict = get_nyt_month_archive(year, str(month))

            # get a dict of article urls of interest:
            article_url_dicts = get_nyt_url_dicts(json_dict)

            arch_file.write(json.dumps(article_url_dicts))
            arch_file.write('\n')


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
                    logging.info(f'Getting page to scrape: {url}')

                    if validators.url(url) == True:
                        response = requests.get(url)
                        # If the response was successful, no Exception will be raised
                        response.raise_for_status()
                    else:
                        logging.exception(f'url is invalid!  url:  {url}')
                        continue
                except requests.exceptions.HTTPError as http_err:
                    logging.exception(f'HTTP error occurred: {http_err} - url: {url}')  # Python 3.6
                    continue
                except Exception as err:
                    logging.exception(f'Other error occurred: {err} - url: {url}')  # Python 3.6
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
                    
                    # now sleep a random period of time
                    time.sleep(random.choice(sleep_sequence))
 
# we will remove pickling as it's redundant
#        pickle.dump(article_collection, pickle_file)






if __name__ == '__main__':
    #
    # test our script with constrained API call and
    main()
