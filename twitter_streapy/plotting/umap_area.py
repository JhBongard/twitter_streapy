
import preprocessing.prepro as pp
from use import use_embedding

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import plotly
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def get_tweets_from_cluster_list(cluster_list):
    tweet_list = []
    for i in range(0, len(cluster_list)):
        tweet_list.extend([y for y in cluster_list[i]["tweets"]])
    return tweet_list


def get_embeddings_from_cluster_list(cluster_list):
    embedding_list = []
    for i in range(0, len(cluster_list)):
        embedding_list.extend([y for y in cluster_list[i]["fv"]])
    return embedding_list


def get_cluster_id_list(cluster_list):
    cluster_id_list = []
    for cluster in range(0, len(cluster_list)):
        for y in range(0, len(cluster_list[cluster]["tweet_ids"])):
            cluster_id_list.append(cluster_list[cluster]["cluster_id"])
    return cluster_id_list



def umap_2D_plotting(tweets, embeddings, clusterlabels, filename):
    grams = pp.get_cluster_top_gram(tweets, clusterlabels)

    # UMAP Dimension reduction
    reducer= umap.UMAP(#n_neighbors=,
                        n_components=2,
                        #min_dist=,
                        #metric=
                        )
    reduced_embed = reducer.fit_transform(embeddings)

    # Plotly
    fig = make_subplots(rows=1, cols=1)
    clusters = np.unique(clusterlabels)
    for cluster,topgram in zip(clusters,grams):
        fig.add_trace(
            go.Scatter(
                x=reduced_embed[clusterlabels == cluster][:, 0],
                y=reduced_embed[clusterlabels == cluster][:, 1],
                mode="markers",
                hovertext=np.array([i["text"] for i in tweets])[clusterlabels == cluster],  # https://plotly.com/python/hover-text-and-formatting/
                name="Cluster "+str(cluster)+" "+str(topgram),
                marker=dict(
                    size=4,
                    color=cluster,
                    colorscale="Portland", #https://plotly.com/python/builtin-colorscales/
                    opacity=0.7
                    )
            )
        )
    fig.update_layout(template="plotly_dark") #https://plotly.com/python/templates/
    fig.update_layout(showlegend=True, legend_title='<b> Legend </b>')
    plotly.offline.plot(fig, filename=filename)




def umap_3D_plotting(tweets, embeddings, clusterlabels, filename):

    grams = pp.get_cluster_top_gram(tweets, clusterlabels)

    reducer = umap.UMAP(  # n_neighbors=,
        n_components=3,
        # min_dist=,
        # metric="cosine"
    )
    reduced_embed = reducer.fit_transform(np.array(embeddings))

    fig = make_subplots(rows=1, cols=1)

    clusters = np.unique(clusterlabels)

    for cluster, topgram in zip(clusters,grams):
        fig.add_trace(
            go.Scatter3d(
                x=reduced_embed[clusterlabels == cluster][:, 0],
                y=reduced_embed[clusterlabels == cluster][:, 1],
                z=reduced_embed[clusterlabels == cluster][:, 2],
                mode="markers",
                hovertext=np.array([i["text"] for i in tweets])[clusterlabels == cluster],
                # https://plotly.com/python/hover-text-and-formatting/
                name="Cluster "+str(cluster)+" "+str(topgram),
                marker=dict(
                    size=4,
                    color=cluster,
                    colorscale="Portland",  # https://plotly.com/python/builtin-colorscales/
                    opacity=0.7
                )
            )
        )
    fig.update_layout(template="plotly_dark")  # https://plotly.com/python/templates/
    # fig.update_layout(showlegend=True, legend_title='<b> Legend </b>')
    plotly.offline.plot(fig, filename=filename)




def area_plot(tweets, clusters, resampling, filename):
    """
    resampling "0.1Min"
    # https://stackoverflow.com/questions/61138732/single-legend-for-plotly-subplot-for-line-plots-created-from-two-data-frames-in
    :param tweets:
    :param cluster_list_file:
    :param resampling:
    :param filename:
    :return:
    """
    cluster_labels = []
    for i in tweet_subset:
        for y in clusters:
            if i["id"] in y["tweet_ids"]:
                cluster_labels.append(y["cluster_id"])

    grams = pp.get_cluster_top_gram(tweets, cluster_labels)

    #clusters = np.unique(cluster_labels)
    fig = go.Figure()
    for cluster, topgram in zip(clusters, grams):
        tweets_time = [tweet["created_at"] for tweet in tweets if tweet["id"] in cluster["tweet_ids"]]
        count = dict(Counter(tweets_time))
        count_df = pd.DataFrame(count.items(), columns=["date", "count"])
        count_df["date"] = pd.to_datetime(count_df["date"])
        count_df = count_df.set_index("date")
        count_df = count_df["count"].resample(resampling).sum()

        fig.add_trace(go.Scatter(x=count_df.index,
                                 y=count_df,
                                 stackgroup='one',
                                 name="Cluster "+str(cluster["cluster_id"])+" "+str(topgram)))


    fig.update_layout(template="plotly_dark")
    plotly.offline.plot(fig, filename=filename)






def area_2D(clusters, tweets, resampling, filename):

    # get & subset tweets
    #tweets = tw_ndjson.import_tweets(tweets_file)
    # compare two lists with set
    tweet_subset = set([i["id"] for i in tweets]) & set([item for sublist in [y["tweet_ids"] for y in clusters] for item in sublist])
    tweet_subset = [i for i in tweets if i["id"] in tweet_subset]


    # get embeddings
    print("embeddings")
    #embeddings = [use.use_vector([i["text"]]) for i in tweet_subset]

    embeddings = use_embedding(tweet_subset)

    # UMAP Dimension reduction
    print("reducer")
    reducer = umap.UMAP(  # n_neighbors=,
        n_components=2,
        # min_dist=,
        # metric=
    )
    reduced_embed = reducer.fit_transform(embeddings)

    # get clusterlabels
    clusterlabels = []
    for i in tweet_subset:
        for y in clusters:
            if i["id"] in y["tweet_ids"]:
                clusterlabels.append(y["cluster_id"])
    len(clusterlabels)
    len(tweet_subset)
    clusters_unique = np.unique(clusterlabels)
    # sort unique cluster labels list by their size
    #clusters_unique=[x for _, x in sorted(zip([i["size"] for i in clusters], clusters_unique), reverse=True)]


    # get clusters topgram
    print("n-grams")
    grams = pp.get_cluster_top_gram(tweet_subset, clusterlabels)

    # Plotly
    fig = make_subplots(rows=2, cols=1)#, subplot_titles=['Plot 1','Plot 2'])

    # 2D plot

    colors = list(plt.cm.Dark2.colors)
    len(colors)
    for i in range(0,10):
        colors.extend(colors)

    colors=iter(colors)
    #colors = iter([plt.cm.Paired(i) for i in range(20)])
    #https://stackoverflow.com/questions/59149279/matplotlib-selecting-colors-within-qualitative-color-map
    #https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib



    cluster_id = clusters_unique[0]
    topgram = grams[0]


    for cluster_id, topgram in zip(clusters_unique, grams):

        color=next(colors)
        color = "rgb("+str(color[0])+","+ str(color[1])+","+str(color[2])+")"
        fig.add_trace(
            go.Scatter(
                x=reduced_embed[clusterlabels == cluster_id][:, 0],
                y=reduced_embed[clusterlabels == cluster_id][:, 1],
                mode="markers",
                legendgroup=str(cluster_id),
                hovertext=np.array([i["text"] for i in tweet_subset])[clusterlabels == cluster_id],
                # https://plotly.com/python/hover-text-and-formatting/
                name="Cluster " + str(cluster_id) + " " + str(topgram),
                marker=dict(
                    size=4,
                    color=color,
                    #color=cluster_id,
                    #colorscale=colors[0],  # https://plotly.com/python/builtin-colorscales/
                    opacity=0.7
                )
            ),
            row=1, col=1
        )

        cluster = [i for i in clusters if i["cluster_id"]==cluster_id]
        tweets_time = [tweet["created_at"] for tweet in tweet_subset if tweet["id"] in cluster[0]["tweet_ids"]]
        count = dict(Counter(tweets_time))
        #count_df = pd.DataFrame(count.items(), columns=["date", "count"])
        count_df = pd.DataFrame([count]).T

        count_df['date'] = pd.to_datetime(count_df.index)
        count_df.columns = ['count', 'date']
        count_df = count_df.set_index("date")
        count_df = count_df["count"].resample(resampling).sum()

        fig.add_trace(
            go.Scatter(
                x=count_df.index,
                y=count_df,
                stackgroup='one',
                legendgroup=str(cluster_id),
                showlegend=False,
                name="Cluster " + str(cluster[0]["cluster_id"]) + " " + str(topgram),
                line=dict(width=0.7, color=color)
            ),
            row=2, col=1
        )

    fig.update_layout(template="plotly_dark")  # https://plotly.com/python/templates/
    fig.update_layout(showlegend=True, legend_title='<b> Legend </b>')
    plotly.offline.plot(fig, filename=filename)


