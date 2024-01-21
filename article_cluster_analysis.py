"""
Christian Peterson
Article Cluster Analysis (based on keyword "israel")
(Utilizes K-Means Clustering Algorithm)
[November/December 2023]
"""

import requests
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator # to find the "elbow"/"knee"
import matplotlib.pyplot as plt
from collections import Counter

KEY_FILE = "___________"
API_BASE_URL = "https://newsapi.org/v2/everything"
CURVE_TYPE = "convex"
DIRECTION_TYPE = "decreasing"



def read_api_key(filename):
    """
    Parameters: the file name string (filename).

    Returns: the API key (string) located in a file.

    Does: reads in a specific file that contains one line only. That line must 
    include a specific API key, and nothing more or less. This function may not 
    be applicable on your end the key can be made a constant, etc.
    """

    with open(filename, "r") as infile:
        api_key = infile.readline()
    
    return api_key



def get_json(api_root_url, search_query, identifier, 
             dates, lang, api_key):
    """
    Parameters: the root URL for use specified in the API's documentation 
    (api_root_url), a string that represents the search query, a category term 
    that specifies how the API's returned results are to be stored (identifier), 
    dates that the results should be within (dates), a specific language that 
    the results need to be in (lang), and the API key (api_key).

    Returns: output/results, in JSON format, that is returned when the API is 
    called using `requests`.

    Does: keeps adding on to the root URL to "craft" a specific search query 
    (forming a large URL with many search parameters). The search query will 
    return tailored results, which is what the API is supposed to do, after the 
    URL is called using the `requests` library.
    """

    root = f"{api_root_url}?"

    query_format = f"q={search_query}"

    date_format = f"&from={dates[0]}&to={dates[1]}"

    lang_format = f"&language={lang}"

    id_format = f"&sortBy={identifier}"

    key_format = f"&apiKey={api_key}"


    url_params = f"{root}{query_format}{date_format}{lang_format}{id_format}\
        {key_format}"

    response = requests.get(url_params)
    json_output = response.json()

    return json_output



def create_df(json_file, key1):
    """
    Parameters: a JSON formatted object (json_file), and a key (key1).

    Returns: a Pandas DataFrame that has been filled with data from the JSON 
    object.

    Does: Uses json_normalize() to create a DataFrame from a JSON object passed 
    a certain point: a key is defined, creating a specific path to the data 
    that one would want in a DataFrame. The function is easily mutable where 
    one could add more keys onto it (extend the path to their needed data).
    """

    df = pd.json_normalize(json_file[key1])

    return df



def tf_idf(text_col):
    """
    Parameters: a Pandas Series (text_col).

    Returns: a tuple of a TF-IDF fitted model and the features of the 
    model/object.

    Does: uses Scikit-learn's TfidfVectorizer library to compute TF-IDF scores 
    for every word in the corpus (all of the documents in `text_col`), creating 
    an object, in which that object is fitted to produce a model and the feature 
    columns' names are returned. 
    """

    object = TfidfVectorizer(stop_words = "english")

    model = object.fit_transform(text_col)

    features = object.get_feature_names_out()

    return (model, features)



def calc_pca(num_components):
    """
    Parameters: the number of components that the data will be reduced to 
    (num_components).

    Returns: a PCA object.

    Does: uses Scikit-learn's PCA function to create an object for 
    dimensionality reduction based on a specific number of components.
    """

    pca_obj = PCA(n_components = num_components)

    return pca_obj



def reduce_dimensionality(pca_obj, tf_idf_model):
    """
    Parameters: an unfitted PCA object (pca_obj), a fitted TF-IDF model 
    (tf_idf_model).

    Returns: a DataFrame in which dimensionality reduction has been performed.

    Does: performs dimensionality reduction on a TF-IDF model that's converted 
    to an array. The abundance of insignificant data is mitigated and the 
    results are filled into a DataFrame. 
    """

    reduced_array = pca_obj.fit_transform(tf_idf_model.toarray())

    reduced_df = pd.DataFrame(reduced_array)

    return reduced_df



def k_means(k_value, X_data):
    """
    Parameters: an int number of clusters to assign data values to (k_value), 
    all features data that is used for fitting the object (X_data).

    Returns: an unfitted K Means Clustering object, the labels of each data 
    point (which cluster it "belongs" to), the locations of the clusters's 
    centroids in array format, and the value of inertia per the specific k_value.

    Does: uses Scikit-learn's KMeans function to create a K Means Clustering 
    Object, fit that object with all data (no targets/labels are present b/c of 
    Unsupervised Learning). From the fitted model, the cluster centroid 
    locations, inertia value, and assigned clusters for each data value can be 
    extracted.
    """

    kmeans_obj = KMeans(n_clusters = k_value,
                        init = "k-means++",
                        n_init = "auto",
                        random_state = 0)
 
    kmeans_model = kmeans_obj.fit(X_data)

    # Centroid locations
    centroid_array = kmeans_model.cluster_centers_

    inertia_float = kmeans_model.inertia_

    # Labels - the index of each sample's cluster (assigned clusters for each point)
    cluster_sample = kmeans_model.labels_ 

    return (kmeans_obj, cluster_sample, centroid_array, inertia_float)



def find_elbow(x_vals, y_vals, curve_type, dir):
    """
    Parameters: a list of x coordinates (x_vals), a list of
    y coordinates (y_vals), a type of plotted curve (curve_type),
    a direction of that curve (dir).

    Returns: a tuple of the x and y coordinate where the elbow is located.

    Does: uses KneeLocator from `kneed` to find the elbow on the plotted 
    Inertia graph. The x and y coordinates, for that specific point are 
    extracted and returned. 

    Note: one should have a sense of what the graph already looks like,
    as the type of curve and its direction need to be specified.
    """

    kneedle = KneeLocator(x = x_vals, y = y_vals, 
                        curve = curve_type, direction = dir)
    
    x_coordinate = kneedle.elbow
    y_coordinate = kneedle.elbow_y

    return (x_coordinate, y_coordinate)



def optimal_k(df, k_values):
    """
    Parameters: a DataFrame (df) and a list of k_values/the number of clusters 
    (k_values).

    Returns: a tuple of the "optimal" K Value and all of the Inertia values 
    (the Inertia values for all K).

    Does: uses k_means() to find the unfitted K Means Clustering object, 
    cluster indices (assigned clusters), centroid locations, and inertia value 
    for each K Value. find_elbow() is used to find the elbow of the Inertia 
    values for ALL K Values, in which the elbow's K Value is the optimal/ideal 
    value. This K Value is found & returned along with all inertia values (to be 
    used with plotting, later on). 

    Note: inertia values are minorly checked with `%.2f` to mitigate the
    impact of changing decimal values passed the hundreths place. Please
    see my comment on EdStem (https://edstem.org/us/courses/41812/discussion/3819167)
    for further reference or email me.
    """

    inertia_values = []

    for k_val in k_values:
        kmeans_obj, indices_sample, \
            centroid_locations, inertia_val = k_means(k_val, df)
        
        inertia_val = float("%.2f" % inertia_val)        
        inertia_values.append(inertia_val) # prevent randomness from changing decimal 
    
    # returns tuple of (x = k value, y = inertia value)
    ideal_inertia = find_elbow(k_values, inertia_values, 
                               CURVE_TYPE, DIRECTION_TYPE)

    ideal_k = ideal_inertia[0]

    return (ideal_k, inertia_values)



def find_counts(labels):
    """
    Parameters: a list of labels/targets (labels).

    Returns: a dictionary with each unique value
    as the KEY, and the corresponding count of 
    each unique value as the VALUES.

    Does: uses Python's Counter function to
    isolate a list's unique values and obtain their counts.
    """

    counts = Counter(labels)

    return counts



def create_plot(x_data, y_data, 
                    x_lbl, y_lbl, title,
                    line_plot = "False",
                    colors_lst = None,
                    cluster_labels = None,
                    spec_point = None,
                    point_lbl = None):
    """
    Parameters: a list of x coordinates/independent variables (x_data), a list 
    of y coordinates/dependent variables (y_data), labels for the x and y axis 
    (x_lbl & y_lbl), a string to define if the type of graph needed is a line 
    plot (line_plot), a list of colors for the plotted data points (colors_lst), 
    an independent x value for one single point (spec_point), a label for that
    singular plotted point (point_lbl).

    Returns: plots either a line plot or scatterplot based on the string 
    "boolean" parameter line_plot. 

    Does: either plots a line plot or scatterplot. For the line plot, an 
    additional point is also marked because of its significance (in this HW -- 
    optimal K and its inertia value). No additional points are specified in the 
    scatterplot, but labels are defined to differentiate the cluster assignments
    using a dictionary that keeps track of all points, and their features, in each
    cluster (technically - each cluster is being plotted). After this selection, 
    additional features are added to the plot to enhance the visualization.
    """

    if line_plot == "True":

        plt.plot(x_data, y_data, marker = "o")
        plt.xticks(x_data)

        # plot the elbow and add its coordinate, might be unintentionally rounding
        plt.plot(spec_point, y_data[spec_point - 1],
                label = point_lbl, color = "red", marker = "X")
        
        point_coordinate = f"({spec_point},{round(y_data[spec_point], 3)})"

        plt.text(spec_point, y_data[spec_point],
                s = point_coordinate)
    

    if line_plot == "False":

        points_per_cluster = {}
        
        for x_value, y_value, color, label in zip(x_data, y_data, 
                                                  colors_lst, 
                                                  cluster_labels):

            if (color, label) not in points_per_cluster:
                points_per_cluster[(color, label)] = {"x": [], "y": []}
        
            points_per_cluster[(color, label)]["x"].append(x_value)
            points_per_cluster[(color, label)]["y"].append(y_value)

        for (color, label), data in points_per_cluster.items():
            plt.scatter(data["x"], data["y"], color = color, label = label) 

    # Additional features
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.legend()

    return plt.show()



def assign_colors(array, color_types):
    """
    Parameters: a NumPy array (array), a list of colors (color_types).

    Returns: a list containing a specific corresponding color based on each 
    unique value in the array.

    Does: iterates through each value in the array. Each value in the array is 
    given a specific and unique color. In this HW, this is used to assign colors 
    for each cluster (to differentiate them and enhance the plots).

    Condition: the `label` variable must be an int that represents a label or 
    target (assigned color) and `color_types` must have a color per each unique 
    label/target (i.e. this list cannot be shorter than the number of clusters). 
    """

    ordered_colors = []

    for label in array:
        assigned_color = color_types[label]
        ordered_colors.append(assigned_color)

    return ordered_colors



def main():
    # Get the API Key for NewsAPI by reading a stored file on personal device
    api_key = read_api_key(KEY_FILE)

    # Get the JSON output from the API
    query = "israel"
    sort_by = "relevancy"
    dates = ("2023-11-10", "2023-11-21")
    language = "en"
    json_data = get_json(API_BASE_URL, query, sort_by, dates, 
                         language, api_key)

    # Create a DataFrame containing all the results from the API for each news reference
    articles_df = create_df(json_data, "articles")
    
    # Obtain the TF IDF model for every article's content, create a new 
    # DataFrame with all of the scores, and then apply PCA to reduce dimensionality
    tf_idf_model, feature_names = tf_idf(articles_df["content"])
    
    # Reduce the dimensionality of the data
    pca_model = calc_pca(2)
    reduced_scores_df = reduce_dimensionality(pca_model, tf_idf_model)

    # Find the K-Value for which its inertia = an elbow
    k_values = [i for i in range(1, 21)]
    optimal_k_value, each_k_inertia = optimal_k(reduced_scores_df, k_values)


    # [Q1] - What keyword(s) did you use in your query?
    print(f"KEYWORD: {query}\n")
    

    # [Q2] - After PCA has been applied, if number of components is 2, 
    # what are the two values of the first article in your dataset?
    first_values = reduced_scores_df.iloc[0]
    print(f"FIRST ARTICLE's VALUES: \n{first_values}\n")


    # [Q3] - What value of k is optimal based on inertia?
    print(f"OPTIMAL K: {optimal_k_value}\n") 


    # [Q4] - Once k has been optimized, how many articles are in each cluster?
    # Find the labels for the samples, and then obtain a unique count for each cluster
    kmeans_obj, cluster_per_sample, centroid_locations, \
        inertia = k_means(optimal_k_value, 
                        reduced_scores_df)

    counts_dct = find_counts(cluster_per_sample)
    for k,v in counts_dct.items():
        print(f"Cluster Num: {k}, Count: {v}\n")


    # [P1] - Scatterplot after PCA has been applied 
    # Define labels and a title, obtain the coordinates for the points matching 
    # up with each cluster, assign a unique color for each and every cluster,
    # and use create_plot() to create a SCATTERPLOT

    x_lbl = "PCA1"
    y_lbl = "PCA2"
    title = f"Data Points of Keyword '{query}' and Their Respective Clusters"

    # 2 arrays
    x_data = reduced_scores_df[0]
    y_data = reduced_scores_df[1]

    # Will be associated with x_data and y_data (the data points) due to sizing
    color_types = ["red", "blue", "green", "orange", "black"]
    colors_lst = assign_colors(cluster_per_sample, color_types)

    clusters_lst = [str(assigned_cluster) for \
                    assigned_cluster in list(cluster_per_sample)]

    create_plot(x_data, y_data, x_lbl, y_lbl, title, "False",
                    colors_lst, clusters_lst)


    # [P2] - A plot showing why you chose the value of k that you consider optimal
    # Uses the optimal_k function to use all of its data but passes `True` to produce a line plot 
    create_plot(k_values, each_k_inertia,
                         "K-Value/Num Clusters", "Inertia", 
                         "K-Value vs Corresponding Inertia",
                         "True", 
                         spec_point = optimal_k_value, 
                         point_lbl = "Inertia Elbow")
    
main()