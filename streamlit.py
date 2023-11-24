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

    @app.route("/check/")
    @cross_origin()
    def check():
        st.write("hi")
        return "This is the backend for filter bubble."

    app.run(port=8888)