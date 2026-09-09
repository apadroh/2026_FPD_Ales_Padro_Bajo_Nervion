import matplotlib.pyplot as plt


def plot_predictions(
    y_true,
    y_pred,
    title
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        y_true,
        label="Real"
    )

    plt.plot(
        y_pred,
        label="Prediction"
    )

    plt.title(title)

    plt.legend()

    plt.show()