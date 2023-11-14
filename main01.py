"""
Implementing a SVM ML technique.
Part 3. Implement defined ML algorithm and get accuracy.

Author: Julia De Geest
"""
import clean
from SVM_phishing import SVMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class Experiment01:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        train_X, train_y, test_X, test_y = Experiment01.load_data("PhishingData.arff")

        phishing_ML = SVMClassifier()

        phishing_ML.fit(X=train_X, y=train_y)
        my_phishing_predictions = phishing_ML.predict(X=test_X)

        target_names = ['class 0', 'class 1']

        print(classification_report(test_y, my_phishing_predictions, target_names=target_names, zero_division=1))
        return test_y, my_phishing_predictions

    @staticmethod
    def load_data(filename="path/to/chess_data.data"):
        """
        Load the data and partition it into testing and training data.
        :param filename: The location of the data to load from file.
        :return: train_X, train_y, test_X, test_y; each as an iterable object
        (like a list or a numpy array).
        """
        # Clean data using the
        clean_data = clean.clean_data(filename)

        X = clean_data.iloc[:, :-1]
        y = clean_data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
        return X_train, y_train, X_test, y_test

    @staticmethod
    def _get_accuracy(pred_y, true_y):
        """
        Calculates the overall percentage accuracy.
        :param pred_y: Predicted values.
        :param true_y: Ground truth values.
        :return: The accuracy, formatted as a number in [0, 1].
        """
        if len(pred_y) != len(true_y):
            raise Exception("Different number of prediction-values than truth-values.")

        number_of_agreements = 0
        number_of_pairs = len(true_y)

        for individual_prediction_value, individual_truth_value in zip(pred_y, true_y):
            if individual_prediction_value == individual_truth_value:
                number_of_agreements += 1

        accuracy = number_of_agreements / number_of_pairs

        return accuracy


if __name__ == "__main__":
    # Run experiment. Prints a classification report from sklearn.metrics and an overall accuracy from _get_accuracy.
    # Experiment01.run()
    y_true, y_pred = Experiment01.run()
    accuracy = Experiment01._get_accuracy(y_true, y_pred)

    print("Overall accuracy is: %s" % accuracy)
