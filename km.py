import numpy as np
import findspark
findspark.init()

findspark.find()
import pyspark
findspark.find()

from pyspark.mllib.clustering import KMeans, GaussianMixture, GaussianMixtureModel
from pyspark import SparkConf, SparkContext


def getTrueValue(line):
    y = np.array([float(x) for x in line.split(',')])
    return y[-1]  # return the last element (at index -1)


def parseLine(line):
    y = np.array([float(x) for x in line.split(',')])
    return y[0:-1]  # drop the last element (at index -1)


def closestCluster(p, centers):
    bestIndex = 0
    minDist = float("+inf")  # minimum distance
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)  # **: exponentiation
        if tempDist < minDist:
            minDist = tempDist
            bestIndex = i
    return bestIndex


def main():
    sc = SparkContext(master="local", appName="K-Means")
    try:
        # csv = sc.textFile(sys.argv[1]) if input via cmd
        csv = sc.textFile("kmeans_data.csv")
    except IOError:
        print('No such file')
        exit(1)

    parsedData = csv.map(parseLine)
    trueValue = csv.map(getTrueValue)
    # print for debugging
    print("number of features: ", len(parsedData.collect()[0]))
    # Build the model (cluster the data), K = 2
    clusters = KMeans.train(
        parsedData, 2, maxIterations=50, initializationMode="random")
    g_clusters = GaussianMixture.train(parsedData, 2)
    centers = clusters.clusterCenters
    # g_centers = g_clusters.clusterCenters
    print("Final k centers:", centers)  # print for debugging purpose
    # print("Final k centers for expectation maximization:", g_centers)

    # for each data point, generate its cluster label:
    predictedLabels = parsedData.map(
        lambda point: closestCluster(point, centers))
    # g_predictedLabels = parsedData.map(lambda point: closestCluster(point, g_centers))
    g_predictedLabels = g_clusters.predict(parsedData)
    results = predictedLabels.collect()
    g_results = g_predictedLabels.collect()
    true = trueValue.collect()
    accuracy_count = 0  # count how many data points having correct labels
    # output in results.txt: i-th row: true label, predicted label for i-th data point:
    g_accuracy_count = 0
    with open("results.txt", "w") as f:
        f.write("true\tpredicted\n")
        for i in range(len(results)):
            f.write(str(true[i]) + "\t" + str(results[i]) + "\n")
            if int(true[i]) == int(results[i]):
                accuracy_count += 1
            if int(true[i]) == int(g_results[i]):
                g_accuracy_count += 1

    accuracy = accuracy_count / len(results)
    g_accuracy = g_accuracy_count / len(results)
    if accuracy < 0.5:  # our predicted label IDs might be opposite
        accuracy = 1 - accuracy
    
    if g_accuracy < 0.5:
        g_accuracy = 1 - g_accuracy
    print("accuracy is :", accuracy)
    print("EM accuracy is : ", g_accuracy)
    sc.stop()


if __name__ == "__main__":
    main()
