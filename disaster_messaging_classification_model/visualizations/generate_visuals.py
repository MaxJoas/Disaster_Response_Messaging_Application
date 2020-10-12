import plotly
from disaster_messaging_classification_model.utils.model_utils import load_data_from_db
from disaster_messaging_classification_model.utils.visuals_utils import plotly_wordcloud


class VisualsGeneration:
    """" module that generates the plotly visualizations """

    def __init__(self):
        self.data = load_data_from_db(set_label="train")

    def generate_plotly_word_cloud_visuals(self):
        """ use plotly wordcloud to generate word cloud visuals """

        # social media messages word cloud
        social_media_messages = " ".join(
            self.data[self.data["genre"] == "social"]["message"]
        )
        social_media_messages = social_media_messages.translate(
            str.maketrans("", "", string.punctuation)
        )
        social_media_layout = plotly_wordcloud(social_media_messages)

        # news messages word cloud
        news_messages = " ".join(self.data[self.data["genre"] == "news"]["message"])
        news_messages = news_messages.translate(
            str.maketrans("", "", string.punctuation)
        )
        news_layout = plotly_wordcloud(news_messages)

        # earthquake related messages
        earthquake_messages = " ".join(
            self.data[self.data["earthquake"] == 1]["message"]
        )
        earthquake_messages = earthquake_messages.translate(
            str.maketrans("", "", string.punctuation)
        )
        earthquake_layout = plotly_wordcloud(earthquake_messages)

        # encode plotly graphs in JSON
        graphs = [social_media_layout, news_layout, earthquake_layout]
        ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON


#  def generate_plotly_visuals(self):

#         # extract data needed for visuals
#         # Message counts of different generes
#         genre_counts = self.data.groupby("genre").count()["message"]
#         genre_names = list(genre_counts.index)

#         # Message counts for different categories
#         cate_counts_df = self.data.iloc[:, 4:].sum().sort_values(ascending=False)
#         cate_counts = list(cate_counts_df)
#         cate_names = list(cate_counts_df.index)

#         # Top keywords in Social Media in percentages
#         social_media_messages = " ".join(
#             self.data[self.data["genre"] == "social"]["message"]
#         )
#         social_media_tokens = tokenize(social_media_messages)
#         social_media_wrd_counter = Counter(social_media_tokens).most_common()
#         social_media_wrd_cnt = [i[1] for i in social_media_wrd_counter]
#         social_media_wrd_pct = [
#             i / sum(social_media_wrd_cnt) * 100 for i in social_media_wrd_cnt
#         ]
#         social_media_wrds = [i[0] for i in social_media_wrd_counter]

#         # Top keywords in Direct in percentages
#         direct_messages = " ".join(self.data[self.data["genre"] == "direct"]["message"])
#         direct_tokens = tokenize(direct_messages)
#         direct_wrd_counter = Counter(direct_tokens).most_common()
#         direct_wrd_cnt = [i[1] for i in direct_wrd_counter]
#         direct_wrd_pct = [i / sum(direct_wrd_cnt) * 100 for i in direct_wrd_cnt]
#         direct_wrds = [i[0] for i in direct_wrd_counter]

#         # create visuals

#         graphs = [
#             # Histogram of the message genere
#             {
#                 "data": [Bar(x=genre_names, y=genre_counts)],
#                 "layout": {
#                     "title": "Distribution of Message Genres",
#                     "yaxis": {"title": "Count"},
#                     "xaxis": {"title": "Genre"},
#                 },
#             },
#             # histogram of social media messages top 30 keywords
#             {
#                 "data": [Bar(x=social_media_wrds[:50], y=social_media_wrd_pct[:50])],
#                 "layout": {
#                     "title": "Top 50 Keywords in Social Media Messages",
#                     "xaxis": {"tickangle": 60},
#                     "yaxis": {"title": "% Total Social Media Messages"},
#                 },
#             },
#             # histogram of direct messages top 30 keywords
#             {
#                 "data": [Bar(x=direct_wrds[:50], y=direct_wrd_pct[:50])],
#                 "layout": {
#                     "title": "Top 50 Keywords in Direct Messages",
#                     "xaxis": {"tickangle": 60},
#                     "yaxis": {"title": "% Total Direct Messages"},
#                 },
#             },
#             # histogram of messages categories distributions
#             {
#                 "data": [Bar(x=cate_names, y=cate_counts)],
#                 "layout": {
#                     "title": "Distribution of Message Categories",
#                     "xaxis": {"tickangle": 60},
#                     "yaxis": {"title": "count"},
#                 },
#             },
#         ]

#         # encode plotly graphs in JSON
#         ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
#         graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
#         return graphJSON

