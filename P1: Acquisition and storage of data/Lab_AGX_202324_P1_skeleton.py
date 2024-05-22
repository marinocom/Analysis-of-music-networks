import networkx as nx
import pandas as pd
import spotipy
from spotipy . oauth2 import SpotifyClientCredentials


# Marino Oliveros Blanco NIU:1668563
# Pere Mayol Carbonell NIU:1669503


# ------- AUXILIARY FUNCTIONS ------- #
def degree_statistics(graph):
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    
    stats = {
        "in_degree": {
            "min": min(in_degrees),
            "max": max(in_degrees),
            "median": sorted(in_degrees)[len(in_degrees)//2]
        },
        "out_degree": {
            "min": min(out_degrees),
            "max": max(out_degrees),
            "median": sorted(out_degrees)[len(out_degrees)//2]
        }
    }
    return stats
# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """

    id = sp.search(artist_name, type='artist', limit=5)['artists']['items'][0]['id'] # Get the first result

    return id


def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS", 
            out_filename: str = "g.graphml") -> nx.DiGraph:
    
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """

    G = nx.DiGraph()
    visited = set()

    if strategy == "BFS":
        queue = [(seed, None)]  
    else:
        stack = [(seed, None)]  # DFS

    # Crawl the graph
    while len(visited) < max_nodes_to_crawl:
        if strategy == "BFS" and queue:
            current_artist_id, parent_id = queue.pop(0)
        elif strategy == "DFS" and stack:
            current_artist_id, parent_id = stack.pop()
        else:
            break
        
        if current_artist_id not in visited:
            visited.add(current_artist_id)
            try:
                artist_info = sp.artist(current_artist_id)
                G.add_node(current_artist_id, name=artist_info['name'])
                if parent_id:
                    G.add_edge(parent_id, current_artist_id)

                related_artists = sp.artist_related_artists(current_artist_id)['artists']
                for related_artist in related_artists:
                    if strategy == "BFS":
                        queue.append((related_artist['id'], current_artist_id))
                    else:
                        stack.append((related_artist['id'], current_artist_id))
            except Exception as e:
                print(f"Error processing artist {current_artist_id}: {e}")
                continue

    nx.write_graphml(G, out_filename)
    return G


def get_track_data(sp, graphs, output_file):
    """
    Get track data for each visited artist in the graph.

    :param sp: spotipy client object
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.

    """
    tracks = []
    # Get top tracks for each artist in the graph (10 in this case)
    for graph in graphs:
        for node in graph.nodes(data=True):
            artist_id = node[0]
            results = sp.artist_top_tracks(artist_id)
            
            for track in results['tracks']:
                tracks.append({
                    'artist_id': artist_id,
                    'artist_name': track['artists'][0]['name'],
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'album_id': track['album']['id'],
                    'album_name': track['album']['name']
                })

    # Save the data to a csv file so we can correctly output the songs.csv
    df = pd.DataFrame(tracks)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":

    CLIENT_ID = "b6476126405647e2a1f3b51d172f38cf"
    CLIENT_SECRET = "dab080b76dc84114905908984031993a"


    auth_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)

    id1 = search_artist(sp, "Taylor Swift")
    Gb = crawler(sp, id1, 100, strategy='BFS', out_filename="gB.graphml") # 100 nodes/artists
    Gd = crawler(sp, id1, 100, strategy='DFS', out_filename="gD.graphml")

    id2 = search_artist(sp, "Pastel Ghost")
    Hb = crawler(sp, id2, 100, strategy='BFS', out_filename="hB.graphml")

    # Degree statistics
    stats_gb = degree_statistics(Gb)
    stats_gd = degree_statistics(Gd)

    print(f"gB degree stats: {stats_gb}")
    print(f"gD degree stats: {stats_gd}")

    # Track data
    D = get_track_data(sp, [Gb, Gd], "songs.csv")
    
    num_songs = D['track_id'].nunique()
    num_artists = D['artist_id'].nunique()
    num_albums = D['album_id'].nunique()

    print(f"Artists: {num_artists}")
    print(f"Albums: {num_albums}")
    print(f"Songs: {num_songs}")