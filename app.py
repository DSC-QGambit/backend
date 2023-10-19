from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
import json
from views import index
import sys
import json
from time import mktime
import nltk
from datetime import datetime
import feedparser as fp
import newspaper
from newspaper import Article

import requests
from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download('punkt')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS']='Content-Type'

@app.route("/")
@cross_origin()
def pg():
    index()
    return "k"

### NEWS PAPER SCRAPING

data = {}
data["newspapers"] = {}
max_articles_from_single_source_limit = 10

def parse_config(fname):
    # Loads the JSON files with news sites
    with open(fname, "r") as data_file:
        cfg = json.load(data_file)

    for company, value in cfg.items():
        if "link" not in value:
            raise ValueError(f"Configuration item {company} missing obligatory 'link'.")

    return cfg


def _handle_rss(company, value, count, limit):
    """If a RSS link is provided in the JSON file, this will be the first
    choice. If you do not want to scrape from the RSS-feed, just leave the RSS
    attr empty in the JSON file.
    """

    fpd = fp.parse(value["rss"])
    # print(f"Downloading articles from {company}")
    news_paper = {"rss": value["rss"], "link": value["link"], "articles": []}
    for entry in fpd.entries:
        # Check if publish date is provided, if no the article is
        # skipped.  This is done to keep consistency in the data and to
        # keep the script from crashing.
        if not hasattr(entry, "published"):
            continue
        if count > limit:
            break
        article = {}
        article["link"] = entry.link
        date = entry.published_parsed
        article["published"] = datetime.fromtimestamp(mktime(date)).isoformat()
        try:
            content = Article(entry.link)
            content.download()
            content.parse()
        except Exception as err:
            # If the download for some reason fails (ex. 404) the
            # script will continue downloading the next article.
            print(err)
            print("continuing...")
            continue
        article["title"] = content.title
        article["text"] = content.text
        article["keywords"] = content.keywords
        article["top_image"] = content.top_image
        news_paper["articles"].append(article)
        # print(f"{count} articles downloaded from {company}, url: {entry.link}")
        count = count + 1
    return count, news_paper


def _handle_fallback(company, value, count, limit):
    """This is the fallback method that uses the python newspaper library 
    to extract articles if a RSS-feed link is not provided."""

    # print(f"Building site for {company}")
    paper = newspaper.build(value["link"], memoize_articles=False)
    news_paper = {"link": value["link"], "articles": []}
    none_type_count = 0
    for content in paper.articles:
        if count > limit:
            break
        try:
            content.download()
            content.parse()
            content.nlp()
        except Exception as err:
            print(err)
            print("continuing...")
            continue
        # If there is no found publish date the article will be skipped.
        # After 10 downloaded articles from the same newspaper without publish date, the company will be skipped.
        if content.publish_date is None:
            # print(f"{count} Article has date of type None...")
            none_type_count = none_type_count + 1
            if none_type_count > 10:
                # print("Too many noneType dates, aborting...")
                none_type_count = 0
                break
            count = count + 1
            continue
        article = {
            "title": content.title,
            "text": content.text,
            "link": content.url,
            "keywords": content.keywords,
            "top_image": content.top_image,
            "published": content.publish_date.isoformat(),
        }
        news_paper["articles"].append(article)
        print(
            f"{count} articles downloaded from {company} using newspaper, url: {content.url}, keywords:{content.keywords}, top_images:{content.top_image}"
        )
        count = count + 1
        none_type_count = 0
    return count, news_paper


def run(config, limit=4):
    """Take a config object of sites and urls, and an upper limit. Iterate through each news company.
    Write result to scraped_articles.json."""
    for company, value in config.items():
        count = 1
        if "rss" in value:
            count, news_paper = _handle_rss(company, value, count, max_articles_from_single_source_limit)
        else:
            count, news_paper = _handle_fallback(company, value, count, max_articles_from_single_source_limit)
        data["newspapers"][company] = news_paper

    # Finally it saves the articles as a JSON-file.
    try:
        with open("scraped_articles.json", "w") as outfile:
            json.dump(data, outfile, indent=2)
    except Exception as err:
        print(err)

def news():
    """News site scraper."""
    try:
        config = parse_config("NewsPapersTop.json")
    except Exception as err:
        sys.exit(err)
    run(config)

#------------------------------------------------------------
### APIS

### Returns the list of all articles from scraped_articles.json

@app.route("/get-top-news-articles/", methods=['GET'])
def get_top_news_articles():

    news()
    articles_list = []
    with open("scraped_articles.json", "r") as data_file:
        scraped_file = json.load(data_file)

    for papers, eachpaper in scraped_file.items():
        for paper, link_and_articles in eachpaper.items():
            if "link" not in link_and_articles:
                raise ValueError(f"Configuration item {link_and_articles} missing obligatory 'link'.")
            else:
                for article in link_and_articles['articles']:
                    articles_list.append(article)
            
    return jsonify(articles_list)

### Accepts the article the user clicks on and sends it's keywords 
### to the twitter data extraction function

@app.route("/post-selected-news-article/", methods=['GET','POST'])
def post_selected_news_article():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        # print( request.get_json() )
        for keyword in request.get_json()['keywords']:
            print(keyword)
            break
    return jsonify(response_object)

@app.route("/get-selected-news-keywords/", methods=['GET'])
def get_selected_news_keywords():

    keywords_list = []
    with open("scraped_articles.json", "r") as data_file:
        scraped_file = json.load(data_file)

    for comp, paper in scraped_file.items():
        for b, value in paper.items():
            if "link" not in value:
                raise ValueError(f"Configuration item {value} missing obligatory 'link'.")
            else:
                for article in value['articles']:
                    for keyword in article['keywords']:
                        print("hi")
                        keywords_list.append(keyword)
            
    return jsonify(keywords_list)

# @app.route("/get-top-news-keywords/", methods=['GET'])
# def get_top_news_keywords():

#     keywords_list = []
#     with open("scraped_articles.json", "r") as data_file:
#         scraped_file = json.load(data_file)

#     for comp, paper in scraped_file.items():
#         for b, value in paper.items():
#             if "link" not in value:
#                 raise ValueError(f"Configuration item {value} missing obligatory 'link'.")
#             else:
#                 for article in value['articles']:
#                     for keyword in article['keywords']:
#                         keywords_list.append(keyword)
            
#     return jsonify(keywords_list)

@app.route("/get-related-news-from-keywords/", methods=['GET','POST'])
def get_related_news_from_keywords():
    keywords = request.get_json()
    print(f'KEYWORDS: {keywords}')

    config_file = "NewsPapersSearch.json"
    matching_articles = []
    limit = 5 # from each source

    try:
        with open(config_file, 'r') as data_file:
            config = json.load(data_file)

        for site, site_data in config.items():
            count = 0
            print(f'Site: {site_data}')
            site_url = site_data['link']
            response = requests.get(site_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('item') + soup.find_all('article')  # You might need to adjust this selector

                for article in articles:
                    if count > limit:
                        break
                    article_text = article.get_text()
                    # Count the number of matching keywords in the article text
                    num_matching_keywords = sum(keyword.lower() in article_text.lower() for keyword in keywords)
                    
                    # Check if at least 50% of the keywords are common
                    if num_matching_keywords >= len(keywords) / 6:
                        matching_articles.append((site, article_text))
                    # if any(keyword.lower() in article_text.lower() for keyword in keywords):
                    #     matching_articles.append((site, article_text))
                    count = count + 1

    except Exception as e:
        print(f"Error: {e}")

    # model = SentenceTransformer('all-MiniLM-L6-v2')

    # # List of sentences
    # sentences = [
    #     'A man is eating food.',
    #     'A man is eating a piece of bread.',
    #     'The girl is carrying a baby.',
    #     'A man is riding a horse.',
    #     'A woman is playing violin.',
    #     'Two men pushed carts through the woods.',
    #     'A man is riding a white horse on an enclosed ground.',
    #     'A monkey is playing drums.',
    #     'Someone in a gorilla costume is playing a set of drums.'
    # ]

    # # Encode all sentences
    # embeddings = model.encode(sentences)

    # # Choose the fixed article
    # fixed_article_index = 0
    # fixed_article = sentences[fixed_article_index]

    # # Encode the fixed article
    # fixed_article_embedding = embeddings[fixed_article_index]

    # # Compute cosine similarity between the fixed article and all other articles
    # cos_sim_with_fixed_article = util.pytorch_cos_sim(fixed_article_embedding, embeddings)

    # # Add all pairs to a list with their cosine similarity score
    # all_sentence_combinations = []
    # for i in range(len(cos_sim_with_fixed_article)):
    #     if i != fixed_article_index:
    #         all_sentence_combinations.append([cos_sim_with_fixed_article[i], fixed_article_index, i])

    # # Sort list by the lowest cosine similarity score (most dissimilar)
    # all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0])

    # print("Top-5 most dissimilar pairs:")
    # for score, i, j in all_sentence_combinations[:5]:
    #     print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], score))

   


    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask



    #Sentences we want sentence embeddings for
    sentences = ['This framework generates embeddings for each input sentence',
                'Sentences are passed as a list of string.',
                'The quick brown fox jumps over the lazy dog.']

    #Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained("TaylorAI/gte-tiny")
    model = AutoModel.from_pretrained("TaylorAI/gte-tiny")

    #Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return jsonify(matching_articles)

#-----------------------------

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)