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

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download('punkt')

app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
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

def summarise_article(text):
    ## Using SUmy package
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    #Summarize the document with 4 sentences
    summary = summarizer(parser.document, 4)
    summary_concat = ""
    for sentence in summary:
        summary_concat = summary_concat + ' ' + str(sentence)

    return summary_concat

### APIS

### Returns the list of all articles from scraped_articles.json

@app.route("/get-top-news-articles/", methods=['GET'])
@cross_origin()
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
                    article['source'] = str(list(eachpaper.keys())[0])
                    # print('here')
                    article['text']=summarise_article(article['text'])
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
            # print(keyword)
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
    import praw
    from praw.models import MoreComments

    titles = []
    titles.append(request.get_json()) # The first in the list is the queried article title
    ids = [0]

    reddit_read_only = praw.Reddit(client_id="oVMUat1BYXBgQw-ksed5Hg",         # your client id
                                client_secret="CKVIiUs4Ma07bP31gSZ3jt0hMqD0AQ",    # your client secret
                                user_agent="News_Subreddit_Crawler")   # your user agent

    subreddit = reddit_read_only.subreddit("news")

    for post in subreddit.new(limit=20):
        titles.append(post.title)
        ids.append(post.id)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(titles)
    fixed_article_embedding = embeddings[0]
    cos_sim_with_fixed_article = util.pytorch_cos_sim(fixed_article_embedding, embeddings)

    all_sentence_combinations = []
    for i in range(len(cos_sim_with_fixed_article[0])):
        if i != 0:
            all_sentence_combinations.append([cos_sim_with_fixed_article[0][i], 0, i])

    # print(titles)
    # print(all_sentence_combinations)

    index_of_max_value = max(range(len(all_sentence_combinations)), key=lambda i: all_sentence_combinations[i][0])

    print("Index of tuple with highest value:", index_of_max_value)
    print("Tuple with highest value:", all_sentence_combinations[index_of_max_value])

    submission = reddit_read_only.submission(id=ids[index_of_max_value+1])

    post_comments = []

    for comment in submission.comments: #with depth?
        if type(comment) == MoreComments:
            continue

        post_comments.append(comment.body)

    for comment in post_comments:
        print(f'-> {comment}')
    
    public_opinion = "Analysis on the reddit thread about the news '{titles[index_of_max_value+1]}' suggest the following sentiments: \n Positive: {x}"

    return jsonify(0)

#-----------------------------

if __name__ == '__main__':
    app.run(threaded=True, port=5000)