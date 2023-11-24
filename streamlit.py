import streamlit as st

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    from flask import Flask, request, jsonify
    from flask_cors import CORS, cross_origin
    import json
    from views import index
    import sys
    from time import mktime
    import nltk
    from datetime import datetime
    import feedparser as fp
    import newspaper
    from newspaper import Article

    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline
    import praw
    from praw.models import MoreComments
    from sklearn.cluster import KMeans

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    nltk.download('punkt')

    app = Flask(__name__)
    # CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
    CORS(app, resources={r"/*": {"origins": "https://filterbubble.netlify.app/"}})
    app.config['CORS_HEADERS']='Content-Type'

    @app.route("/")
    @cross_origin()
    def pg():
        index()
        return "This is the backend for filter bubble."

    ### NEWS PAPER SCRAPING

    data = {}
    data["newspapers"] = {}
    max_articles_from_single_source_limit = 4

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
        news_paper = {"rss": value["rss"], "link": value["link"], "articles": []}
        for entry in fpd.entries:
            # Check if publish date is provided, if no the article is
            # skipped.  This is done to keep consistency in the data and to
            # keep the script from crashing.
            if (not hasattr(entry, "published")):
                continue
            if ((not hasattr(entry, "top_image")) or (entry["top_image"] == "")):
                continue
            if ((not hasattr(entry, "title")) or (entry["title"] == "")):
                continue
            if ((not hasattr(entry, "text")) or (entry["text"] == "")):
                continue
            if ((not hasattr(entry, "keywords")) or (entry["keywords"] == [])):
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
            print(f"{count} articles downloaded from {company}, title: {entry.title}")
            count = count + 1
        return count, news_paper


    def _handle_fallback(company, value, count, limit):
        """This is the fallback method that uses the python newspaper library 
        to extract articles if a RSS-feed link is not provided."""

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
            if content.publish_date is None or content.title == "" or content.top_image == "" or content.text == "" or content.keywords == []:
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

            existing_titles = [existing_article["title"] for existing_article in news_paper["articles"]]
            if article["title"] not in existing_titles:
                news_paper["articles"].append(article)
                print(
                    f"{count} articles downloaded from {company} using newspaper, title: {content.title}"
                )
                count = count + 1
            none_type_count = 0
        return count, news_paper


    def run(config):
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

    ### Helper functions

    import re

    def remove_links(input_string):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        matches = re.findall(url_pattern, input_string)
        for match in matches:
            input_string = input_string.replace(match, '')

        input_string = input_string.replace('[deleted] ', '')
        return input_string

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
                        article['source'] = paper
                        articles_list.append(article)
                
        return jsonify(articles_list)


    @app.route("/get-article-summary/", methods=['GET', 'POST'])
    @cross_origin()
    def get_article_summary():
        summarizer = pipeline("summarization", model="mrm8488/t5-base-finetuned-summarize-news", tokenizer="mrm8488/t5-base-finetuned-summarize-news", framework="pt")
        input = request.get_json()
        if len(input) >= 500:
            max_len = 500
            input = input[:500]
        else:
            max_len = len(input) - 1
        
        summary = summarizer(input, min_length=5, max_length=max_len)

        summary = summary[0]['summary_text']
        last_period_index = summary.rfind('.')
        if last_period_index != -1:
            summary = summary[:last_period_index + 1]
                
        return jsonify(summary)

    ### Accepts the article the user clicks on and sends it's keywords 
    ### to the twitter data extraction function

    @app.route("/get-related-articles/", methods=['POST'])
    def get_related_articles():
        reddit_read_only = praw.Reddit(client_id="oVMUat1BYXBgQw-ksed5Hg",
                                    client_secret="CKVIiUs4Ma07bP31gSZ3jt0hMqD0AQ",
                                    user_agent="News_Subreddit_Crawler")

        related_reddit_posts = []
        limit = 3
        count = 0
        search_query = ""

        for keyword in request.get_json()['keywords']:
            search_query += " " + keyword
        
        listing = reddit_read_only.subreddit("news").search(search_query, time_filter = 'month')
        
        for id in listing:
            post = reddit_read_only.submission(id=id)
            
            if "megathread" in post.title.lower():
                continue
            
            related_reddit_posts.append({"title": post.title, "url": post.url})
            count += 1

            if(count == limit):
                break

        return jsonify(related_reddit_posts)


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


    @app.route("/get-public-opinion-from-reddit/", methods=['GET','POST'])
    def get_public_opinion_from_reddit():
        titles = []
        ids = []
        urls = []

        reddit_read_only = praw.Reddit(client_id="oVMUat1BYXBgQw-ksed5Hg",         # your client id
                                    client_secret="CKVIiUs4Ma07bP31gSZ3jt0hMqD0AQ",    # your client secret
                                    user_agent="News_Subreddit_Crawler")   # your user agent

        subreddit = reddit_read_only.subreddit("news")

        for post in subreddit.new(limit=200):
            titles.append(post.title)
            ids.append(post.id)
            urls.append(post.url)

        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embeddings = model.encode(request.get_json(), convert_to_tensor=True)
        corpus_embeddings = model.encode(titles, convert_to_tensor=True)
        top_matches = util.semantic_search(query_embeddings, corpus_embeddings, top_k=2)

        all_public_sentiments = []
        for match in top_matches[0]:
            submission = reddit_read_only.submission(id=ids[match['corpus_id']])
            post_comments = []

            for comment in submission.comments: # Without depth
                if type(comment) == MoreComments:
                    continue

                if(comment.body != '[removed]'):
                    post_comments.append(comment.body)

            classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english") # return_all_scores

            positive_sum, negative_sum, neutral_sum = 0, 0, 0
            public_sentiments = {'source': titles[match['corpus_id']], 'url': urls[match['corpus_id']], 'pos': 0.0, 'neg': 0.0, 'neu': 0.0,
                                'summary': []}

            if(len(post_comments) == 0):
                all_public_sentiments.append(public_sentiments)
                continue

            for comment in post_comments:
                comment = request.get_json()
                if len(comment) >= 512:
                    comment = comment[:512]

                if(classifier(comment)[0]['label']=='POSITIVE'):
                    positive_sum += classifier(comment)[0]['score']
                elif(classifier(comment)[0]['label']=='NEGATIVE'):
                    negative_sum += classifier(comment)[0]['score']
                else:
                    neutral_sum += classifier(comment)[0]['score']

            public_sentiments['pos'] = positive_sum / len(post_comments)
            public_sentiments['neg'] = negative_sum / len(post_comments)
            public_sentiments['neu'] = neutral_sum / len(post_comments)

            ### K-Means

            post_comments_embeddings = model.encode(post_comments)
            if len(post_comments)>2:
                num_clusters = 2
            else:
                num_clusters = len(post_comments)

            clustering_model = KMeans(n_clusters=num_clusters, n_init="auto")
            clustering_model.fit(post_comments_embeddings)
            cluster_assignment = clustering_model.labels_

            clustered_sentences = [[] for i in range(num_clusters)]
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                clustered_sentences[cluster_id].append(post_comments[sentence_id])

            summarizer = pipeline("summarization", model="mrm8488/t5-base-finetuned-summarize-news", tokenizer="mrm8488/t5-base-finetuned-summarize-news", framework="pt")

            for i, cluster in enumerate(clustered_sentences):
                cluster_string = ' '.join(cluster)
                cluster_string = remove_links(cluster_string)

                if len(cluster_string) >= 300:
                    max_len = 300
                    cluster_string = cluster_string[:300]
                else:
                    max_len = len(cluster_string) - 1
                summary = summarizer(cluster_string, min_length=5, max_length=max_len)
                summary = summary[0]['summary_text']

                last_period_index = summary.rfind('.')
                if last_period_index != -1:
                    summary = summary[:last_period_index + 1]
                public_sentiments['summary'].append(summary)

            all_public_sentiments.append(public_sentiments)

        return jsonify(all_public_sentiments)

    #-----------------------------

    # if __name__ == '__main__':
    #     # app.run(threaded=True, port=5000)
    #     app.run(threaded=True, debug=False)

    app.run(port=8888)