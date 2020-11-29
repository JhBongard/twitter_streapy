

#from preprocessing import prepro
from preprocessing import synergetic_prepro
from stream_fetching import ndjson_handler
from use import use_embedding

import uuid
import numpy as np
import jsonlines
from tqdm import tqdm
from datetime import datetime
from numba import jit
from numba import prange
import copy



@jit(nopython=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    """
    Numba accelerates numerical algorithms in Python & can reach the speed of C
    """
    assert (u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta

@jit(nopython=True,parallel=True)
def cosine_similarity_numba_parallel(cc: np.ndarray, v: np.ndarray):

    out = np.zeros(cc.shape[0], dtype=np.float32)

    for y in prange(cc.shape[0]):
        u=cc[y]
        #assert (u.shape[0] == v.shape[0])
        uv = 0
        uu = 0
        vv = 0
        for i in prange(u.shape[0]):
            uv += u[i] * v[i]
            uu += u[i] * cc[y][i]
            vv += v[i] * v[i]
        cos_theta = 1
        if uu != 0 and vv != 0:
            cos_theta = uv / np.sqrt(uu * vv)

        out[y] = cos_theta
    return out




def create_cluster(twt):
    cluster = {"cluster_id": uuid.uuid4().int & (1 << 64) - 1,
               "start_time": twt["created_at"],
               "end_time": twt["created_at"],
               "tweet_ids": [twt["id"]],
               "hashtag": twt["hashtag"],
               "mention": twt["mention"],
               "coordinates": [twt["coordinates"]]}

    return cluster



def get_central_centroid_t(fv, cc, idx, new_fv):
    # update fv and cc
    # If there are less than 30 embeddings describing the cluster,
    # add tweet embedding to main_fv and update the central centroid (cc)
    # if len(fv[idx].shape) == 1 or len(fv[idx]) < 30:
    if fv[idx].ndim == 1 or len(fv[idx]) < 30:
        fv[idx] = np.vstack((fv[idx], new_fv))
        if cc.ndim == 1:
            cc = fv[idx].mean(axis=0)
        else:
            cc[idx] = fv[idx].mean(axis=0)

    # If there are more than 30 embeddings describing the clusters central centroid,
    # create a sim_list checking all the similarities for the 30 embeddings compared to the central centroid
    # if the similarity of new_fv with the cc is higher than any of the 30, replace the fv containing the min value
    # update the central centroid
    if len(fv[idx]) >= 30:
        if cc.ndim == 1:
            c_sim_list = [cosine_similarity_numba(cc, i) for i in fv[idx]]
            if min(c_sim_list) < cosine_similarity_numba(cc, new_fv):
                fv[idx][c_sim_list.index(min(c_sim_list))] = new_fv
                cc = fv[idx].mean(axis=0)

        else:
            c_sim_list = [cosine_similarity_numba(cc[idx], i) for i in fv[idx]]
            if min(c_sim_list) < cosine_similarity_numba(cc[idx], new_fv):
                fv[idx][c_sim_list.index(min(c_sim_list))] = new_fv
                cc[idx] = fv[idx].mean(axis=0)

    return fv, cc

def get_central_centroid_c(main_fv, main_cc, new_fv):
    # If there are less than 30 embeddings describing the cluster,
    # add tweet embedding to main_fv and update the central centroid (cc)
    if len(main_fv.shape) == 1 or len(main_fv) < 30:
        main_fv = np.vstack((main_fv, new_fv))
        main_cc = main_fv.mean(axis=0)

    # If there are more than 30 embeddings describing the clusters central centroid,
    # create a sim_list checking all the similarities for the 30 embeddings compared to the central centroid
    # if the similarity of new_fv with the cc is higher than any of the 30, replace the fv containing the min value
    # update the central centroid
    if len(main_fv) >= 30:
        sim_list = [cosine_similarity_numba(main_cc, i) for i in main_fv]
        if min(sim_list) < cosine_similarity_numba(main_cc, new_fv):
            main_fv[sim_list.index(min(sim_list))] = new_fv
            main_cc = main_fv.mean(axis=0)

    return main_fv, main_cc

def append_tweet_to_cluster(cluster_list, idx, tweet, fv, cc, new_fv):
    # update cluster in cluster_list
    cluster_list[idx]["end_time"] = tweet["created_at"]
    cluster_list[idx]["tweet_ids"].append(tweet["id"])

    cluster_list[idx]["hashtag"].extend(tweet["hashtag"])
    cluster_list[idx]["hashtag"] = list(set(cluster_list[idx]["hashtag"]))
    cluster_list[idx]["mention"].extend(tweet["mention"])
    cluster_list[idx]["mention"] = list(set(cluster_list[idx]["mention"]))

    cluster_list[idx]["coordinates"].append(tweet["coordinates"])
    cluster_list[idx]["coordinates"] = [i for i in cluster_list[idx]["coordinates"] if i != None]

    # update fv and cc
    fv, cc = get_central_centroid_t(fv=fv, cc=cc, idx=idx, new_fv=new_fv)

    return fv, cc  #cluster_list is a global list



def crp(tweets, embeddings, sim_threshold):

    twts = copy.deepcopy(tweets)

    # create a cluster list based on the first tweet
    cluster_list = []
    cluster_list.append(create_cluster(twt=twts[0]))
    fv = [embeddings[0]]
    cc = embeddings[0]

    # assign each tweet
    for tw, embed in zip(twts[1:], embeddings[1:]):

        # calculate tweets similarity to each cluster center
        if cc.ndim == 1:
            sim_list = [cosine_similarity_numba(cc, embed)]
        else:
            sim_list = cosine_similarity_numba_parallel(cc, embed)

        # check if the tweet can be assigned to an existing cluster
        if max(sim_list) >= sim_threshold:
            # get the corresponding cluster index
            idx = np.argmax(sim_list)
            fv, cc = append_tweet_to_cluster(cluster_list=cluster_list,
                                             idx=idx,
                                             tweet=tw,
                                             fv=fv,
                                             cc=cc,
                                             new_fv=embed)
        # or create a new cluster
        else:
            cluster_list.append(create_cluster(twt=tw))
            fv.append(embed)
            cc = np.vstack((cc, embed))

    return cluster_list, fv, cc


def append_cluster_to_cluster(main_idx, new_idx,
                              main_list, new_list,
                              main_cc, new_cc,
                              main_fv, new_fv):

    # update main cluster list
    main_list[main_idx]["start_time"] = min([new_list[new_idx]["start_time"], main_list[main_idx]["start_time"]])
    main_list[main_idx]["end_time"] = max([new_list[new_idx]["end_time"], main_list[main_idx]["end_time"]])

    main_list[main_idx]["tweet_ids"].extend(new_list[new_idx]["tweet_ids"])

    main_list[main_idx]["hashtag"].extend(new_list[new_idx]["hashtag"])
    main_list[main_idx]["hashtag"] = list(set(main_list[main_idx]["hashtag"]))

    main_list[main_idx]["mention"].extend(new_list[new_idx]["mention"])
    main_list[main_idx]["mention"] = list(set(main_list[main_idx]["mention"]))

    main_list[main_idx]["coordinates"].extend(new_list[new_idx]["coordinates"])

    # define the new central centroid as the main centroid,
    # das alte wird an das neue angepasst
    # the central centroid of the new cluster is taken but every old fv is checked if its nearer to the new cluster center
    # check if its an ndimensional array or a simple numpy array
    if len(main_fv[main_idx].shape) == 1:
        main_fv[main_idx], main_cc[main_idx] = \
            get_central_centroid_c(main_fv=new_fv[new_idx],
                                 main_cc=new_cc[new_idx],
                                 new_fv=main_fv[main_idx])
    else:
        old_main_fv = main_fv[main_idx]
        for i in range(0,old_main_fv.shape[0]):
            main_fv[main_idx], main_cc[main_idx] = \
                get_central_centroid_c(main_fv=new_fv[new_idx],
                                     main_cc=new_cc[new_idx],
                                     new_fv=old_main_fv[i])


    return main_list, main_cc, main_fv





def merge_cluster_lists(main_list, main_fv, main_cc,
                        new_list, new_fv, new_cc,
                        cluster_threshold, mode):


    if mode =="single":

        for new_idx, n_cc in enumerate(new_cc):
            cos_sim_list = cosine_similarity_numba_parallel(main_cc, n_cc)


            if max(cos_sim_list) >= cluster_threshold:
                main_idx = np.argmax(cos_sim_list)
                main_list, main_cc, main_fv = append_cluster_to_cluster(main_idx=main_idx,
                                          new_idx=new_idx,
                                          main_list=main_list,
                                          new_list=new_list,
                                          main_cc=main_cc,
                                          new_cc=new_cc,
                                          main_fv=main_fv,
                                          new_fv=new_fv)

            else:
                main_list.append(new_list[new_idx])
                main_cc = np.vstack((main_cc, n_cc))
                main_fv.append(new_fv[new_idx])



    if mode == "multiple":

        for n_cc, n_list in zip(new_cc, new_list):
            # for each cluster of the current interval find clusters in the main list which are near
            cluster_chain = {"new_cluster": n_list["cluster_id"],
                             "main_cluster": [main_list[i[0]]["cluster_id"] for i in np.argwhere(cosine_similarity_numba_parallel(main_cc, n_cc) > cluster_threshold)]}

            # if there is no near main_cluster, append it to main_list, main_cc and main_fv
            if cluster_chain["main_cluster"] == []:
                # get idx from id
                new_idx = [i for i, _ in enumerate(new_list) if _["cluster_id"] == n_list["cluster_id"]][0]

                main_list.append(new_list[new_idx])
                main_cc = np.vstack((main_cc, n_cc))
                main_fv.append(new_fv[new_idx])


            else:
                # if there are main clusters to merge, merge all clusters to the first one in the cluster_chain list
                main_idx = [i for i, _ in enumerate(main_list) if _['cluster_id'] == cluster_chain["main_cluster"][0]][0]

                for i in cluster_chain["main_cluster"][1:]:
                    new_idx = [y for y, _ in enumerate(main_list) if _['cluster_id'] == i][0]
                    main_list, main_cc, main_fv = append_cluster_to_cluster(main_idx=main_idx,
                                                                            new_idx=new_idx,
                                                                            main_list=main_list,
                                                                            new_list=main_list,
                                                                            main_cc=main_cc,
                                                                            new_cc=main_cc,
                                                                            main_fv=main_fv,
                                                                            new_fv=main_fv)

                    # remove merged cluster from main
                    del main_list[new_idx]
                    del main_fv[new_idx]
                    main_cc = np.delete(main_cc, new_idx, 0)


                # merge cluster from new
                new_idx = [i for i, _ in enumerate(new_list) if _['cluster_id'] == n_list["cluster_id"]][0]
                main_idx = [i for i, _ in enumerate(main_list) if _['cluster_id'] == cluster_chain["main_cluster"][0]][0]
                main_list, main_cc, main_fv = append_cluster_to_cluster(main_idx=main_idx,
                                                                        new_idx=new_idx,
                                                                        main_list=main_list,
                                                                        new_list=new_list,
                                                                        main_cc=main_cc,
                                                                        new_cc=new_cc,
                                                                        main_fv=main_fv,
                                                                        new_fv=new_fv)

    return main_list, main_cc, main_fv


def export_cluster_list(cluster_list, filename, mode):
    """
    file name should be .ndjson
    :param inactive_id_list:
    :param main_cluster_list:
    :param filename:
    :return:
    """
    subset = []
    for i in cluster_list:
        line = {"cluster_id": i["cluster_id"],
                "start_time": i["start_time"],
                "end_time": i["end_time"],
                # "size": i["size"],
                "tweet_ids": i["tweet_ids"],
                "mention": i["mention"],
                "hashtag": i["hashtag"],
                # "ner": i["ner"],
                "coordinates": [y for y in i["coordinates"] if y != []]}
        subset.append(line)
    with jsonlines.open(filename, mode=mode) as writer:
        writer.write_all(subset)





def exp_del_inactive_clusters(main_cluster_list, main_cc, main_fv,
                              inactive_time, force_closure_time, filename):
    """
    finds, exports and deletes all clusters from the list that
    - are older than x seconds (force_closure_time by start_time)
    - have not received another cluster for x seconds (inactive_time by end_time)

    :param main_cluster_list:
    :return:
    """
    # find clusters which have not received another cluster for x seconds (end_time)
    print("inactive")
    end_time_list = [datetime.strptime(i["end_time"], '%Y-%m-%d %H:%M:%S') for i in main_cluster_list]
    actual_time = max(end_time_list)
    time_delta_list = [actual_time-i for i in end_time_list]
    inactive_idx_list = [idx for idx, item in enumerate(time_delta_list) if item.seconds >= inactive_time]

    # find clusters which are older than x seconds (start_time)
    print("forcing")
    start_time_list = [datetime.strptime(i["start_time"], '%Y-%m-%d %H:%M:%S') for i in main_cluster_list]
    time_delta_list = [actual_time - i for i in start_time_list]
    closure_idx_list = [idx for idx, item in enumerate(time_delta_list) if item.seconds >= force_closure_time]

    #combine both lists
    inactive_idx_list.extend(closure_idx_list)
    inactive_idx_list = list(set(inactive_idx_list))
    inactive_id_list = [main_cluster_list[i]["cluster_id"] for i in inactive_idx_list]

    # export inactive clusters
    print("export")

    export_cluster_list(cluster_list=[i for i in main_cluster_list if i["cluster_id"] in inactive_id_list],
                        filename=filename,
                        mode="a")


    # delete exported clusters
    main_cluster_list = [i for i in main_cluster_list if i["cluster_id"] not in inactive_id_list]
    main_fv = [fv for i, fv in enumerate(main_fv) if i not in inactive_idx_list]
    main_cc = np.delete(main_cc, inactive_idx_list, 0)

    return main_cluster_list, main_cc, main_fv


def supervised_filtering(cluster_list, fv, cc, tweets):


    # get all relevant tweet ids
    rel_tweets = set([i["id"] for i in tweets if i["relevance_confidence"]>=0.7])
    # get cluster idx and cluster ids which should be removed
    del_idx_list=[]
    del_id_list= []
    for idx, c in enumerate(cluster_list):
        if len(set(c["tweet_ids"]).intersection(rel_tweets))==0:
            del_idx_list.append(idx)
            del_id_list.append(c["cluster_id"])

    # get delete other clusters
    cluster_list = [i for i in cluster_list if i["cluster_id"] not in del_id_list]
    fv = [fv for i, fv in enumerate(fv) if i not in del_idx_list]
    cc = np.delete(cc, del_idx_list, 0)

    # check length of supervised vs clustered tweets
    #t=[item for sublist in [c["tweet_ids"] for c in cluster_list] for item in sublist]
    #len(t)
    #len(rel_tweets)

    return cluster_list, fv, cc


def crp_chain(ndjson_file, time_interval, repetitions,
              sim_threshold, cluster_threshold, mode,
              inactive_time, force_closure_time, filename):

    # import tweets
    tweets, count = ndjson_handler.import_interval(ndjson_file=ndjson_file,
                                                   interval=time_interval,
                                                   start_tweet=0)
    print(len(tweets))


    # tweets are already preprocessed
    # preprocessing
    #tweets = [prepro.preprocessing(tweet=i, word_count_min=2) for i in tweets]
    #tweets = [i for i in tweets if i != None]
    #tweets = synergetic_prepro.clean_relevance_data(tweets)
    # embeddings
    embeddings = use_embedding(tweets)

    # crp
    print("crp")
    main_list, main_fv, main_cc = crp(tweets, embeddings, sim_threshold)


    # relevance filtering
    print("filtering")
    # it can be that no clusters are chosen
    main_list, main_fv, main_cc = supervised_filtering(cluster_list=main_list,
                                                       fv=main_fv,
                                                       cc=main_cc,
                                                       tweets=tweets)
    if len(main_list)==0:
        print("Filtering: No cluster was selected")
        input()

    # cluster chain
    for i in tqdm(range(0, repetitions)):
        tweets, count = ndjson_handler.import_interval(
            ndjson_file=ndjson_file,
            interval=time_interval,
            start_tweet=count)
        print(len(tweets))
        #print("preprocessing")
        #tweets = [prepro.preprocessing(tweet=i, word_count_min=2) for i in tweets]
        #tweets = [i for i in tweets if i != None]
        #tweets = synergetic_prepro.clean_relevance_data(tweets)

        print("embedding")
        embeddings = use_embedding(tweets)

        print("crp")
        new_list, new_fv, new_cc = crp(tweets, embeddings, sim_threshold)


        print("crisis lex filtering")
        new_list, new_fv, new_cc = supervised_filtering(cluster_list=new_list,
                                                           fv=new_fv,
                                                           cc=new_cc,
                                                           tweets=tweets)

        print("append_crp")
        main_list, main_cc, main_fv = merge_cluster_lists(main_list=main_list, main_fv=main_fv, main_cc=main_cc,
                                                          new_list=new_list, new_fv=new_fv, new_cc=new_cc,
                                                          cluster_threshold=cluster_threshold, mode=mode)
        print("exp_del")
        main_list, main_cc, main_fv = exp_del_inactive_clusters(main_cluster_list=main_list, main_cc=main_cc, main_fv=main_fv,
                                                                inactive_time=inactive_time,
                                                                force_closure_time=force_closure_time,
                                                                filename=filename)

    export_cluster_list(cluster_list=main_list,
                        filename=filename,
                        mode="a")

    #return main_list



import time
start_time = time.time()
crp_chain(ndjson_file="/media/jb/Volume/03_Supervised/Preprocessed/Event2012_RT_2012-10-24.ndjson",
       time_interval=60*60,
       repetitions=23,#int(86400/3600-1),
       sim_threshold=0.6,
       cluster_threshold=0.9,
       mode="multiple",
       inactive_time=60*60*25,
       force_closure_time=60*60*25,
       filename="/media/jb/Volume/tests/DNN/2012-10-24_60Min_23_0.6_0.9_mltpl_inactive24h_closure24h_V4.ndjson")
print("--- %s seconds ---" % (time.time() - start_time))


