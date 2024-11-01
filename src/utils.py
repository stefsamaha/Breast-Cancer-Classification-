from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the model's performance using accuracy, precision, recall, and F1 score.
    
    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - model_name (str): Name of the model for display in results.

    Returns:
    - metrics (dict): Dictionary containing accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    return metrics

def plot_metrics_comparison(metrics_list):
    """
    Plots a bar chart comparing accuracy, precision, recall, and F1 score for each model.

    Parameters:
    - metrics_list (list): List of dictionaries containing performance metrics for each model.
    """
    # Extract metrics for each model
    models = [metrics["Model"] for metrics in metrics_list]
    accuracies = [metrics["Accuracy"] for metrics in metrics_list]
    precisions = [metrics["Precision"] for metrics in metrics_list]
    recalls = [metrics["Recall"] for metrics in metrics_list]
    f1_scores = [metrics["F1 Score"] for metrics in metrics_list]
    
    # Set up the figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Model Performance Comparison')
    
    # Plot each metric
    ax[0, 0].bar(models, accuracies, color='skyblue')
    ax[0, 0].set_title('Accuracy')
    ax[0, 0].set_ylim(0, 1)
    
    ax[0, 1].bar(models, precisions, color='salmon')
    ax[0, 1].set_title('Precision')
    ax[0, 1].set_ylim(0, 1)
    
    ax[1, 0].bar(models, recalls, color='lightgreen')
    ax[1, 0].set_title('Recall')
    ax[1, 0].set_ylim(0, 1)
    
    ax[1, 1].bar(models, f1_scores, color='orange')
    ax[1, 1].set_title('F1 Score')
    ax[1, 1].set_ylim(0, 1)
    
    # Display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
