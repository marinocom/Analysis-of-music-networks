import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances # for the similarity graph

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    undirected_graph = nx.Graph()
    for u, v in g.edges():
        if g.has_edge(v, u):
            undirected_graph.add_edge(u, v)
    
    nx.write_graphml(undirected_graph, out_filename)
    return undirected_graph


def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    pruned_graph = g.copy()
    nodes_to_remove = [node for node, degree in dict(pruned_graph.degree()).items() if degree < min_degree]
    pruned_graph.remove_nodes_from(nodes_to_remove)

    zero_degree_nodes = [node for node, degree in dict(pruned_graph.degree()).items() if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)
    
    nx.write_graphml(pruned_graph, out_filename)
    return pruned_graph


def prune_low_weight_edges(g: nx.Graph, min_weight: float = None, min_percentile: int = None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    if (min_weight is None and min_percentile is None) or (min_weight is not None and min_percentile is not None):
        raise ValueError("Specify exactly one of min_weight or min_percentile")
    
    else:
        pruned_graph = g.copy()
        if min_weight is not None:
            edges_to_remove = [(u, v) for u, v, w in pruned_graph.edges(data='weight') if w < min_weight]
        else:
            weights = [w for _, _, w in pruned_graph.edges(data='weight')]
            threshold = pd.Series(weights).quantile(min_percentile / 100.0)
            edges_to_remove = [(u, v) for u, v, w in pruned_graph.edges(data='weight') if w < threshold]
        
        pruned_graph.remove_edges_from(edges_to_remove)
        
        zero_degree_nodes = [node for node, degree in dict(pruned_graph.degree()).items() if degree == 0]
        pruned_graph.remove_nodes_from(zero_degree_nodes)
        
        if out_filename:
            nx.write_graphml(pruned_graph, out_filename)
        return pruned_graph


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    mean_features_df = tracks_df.groupby('artist_id').mean().reset_index()
    artist_names = tracks_df[['artist_id', 'artist_name']].drop_duplicates()
    mean_features_df = pd.merge(mean_features_df, artist_names, on='artist_id', how='left')
    return mean_features_df


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """


    features = artist_audio_features_df.drop(['artist_id', 'artist_name'], axis=1).values
    if similarity == 'cosine':
        similarity_matrix = cosine_similarity(features)
    elif similarity == 'euclidean':
        similarity_matrix = euclidean_distances(features)
    else:
        raise ValueError("Unsupported similarity metric. Use 'cosine' or 'euclidean'.")
    
    similarity_graph = nx.Graph()
    num_artists = len(artist_audio_features_df)
    for i in range(num_artists):
        for j in range(i + 1, num_artists):
            similarity_graph.add_edge(
                artist_audio_features_df.loc[i, 'artist_name'],
                artist_audio_features_df.loc[j, 'artist_name'],
                weight=similarity_matrix[i, j]
            )
    
    if out_filename:
        nx.write_graphml(similarity_graph, out_filename)
    return similarity_graph



if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    
    # Load the directed graphs
    gB = nx.read_graphml('P1: Acquisition and storage of data/gB.graphml')
    gD = nx.read_graphml('P1: Acquisition and storage of data/gD.graphml')
    
    # Task 6(a) - Generate undirected graphs with bidirectional edges
    gB_prime = retrieve_bidirectional_edges(gB, 'gBp.graphml')
    gD_prime = retrieve_bidirectional_edges(gD, 'gDp.graphml')
    
    # Load the song data
    tracks_df = pd.read_csv('P1: Acquisition and storage of data/songs.csv')
    
    # Task 6(b) - Generate similarity graph from average audio features
    artist_mean_audio_features_df = compute_mean_audio_features(tracks_df)
    gw = create_similarity_graph(artist_mean_audio_features_df, 'cosine', 'gw.graphml')
    
    # Display resulting graphs (for debugging purposes)
    print(f"gB_prime: {gB_prime}")
    print(f"gD_prime: {gD_prime}")
    print(f"gw: {gw}")